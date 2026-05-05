#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# train.py
import os, time, re, torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda import amp
from torch.utils.data import DataLoader

from preprocess import get_folds
from utils import seed_torch
from augmentation import get_transforms
from dataset import TrainDataset
from models import SwinIVTModel

from preprocess_multi_baseline import build_fused_full_dataframe, build_video2fold_from_schema

import numpy as np
import pandas as pd # pandas import for CSV load
from tqdm.auto import tqdm
import cv2
from hydra.utils import to_absolute_path

cv2.setNumThreads(0)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

try:
    import torch.multiprocessing as mp
    available = set(mp.get_all_sharing_strategies())
    if "file_system" in available:
        mp.set_sharing_strategy("file_system")
    elif "file_descriptor" in available:
        mp.set_sharing_strategy("file_descriptor")
except Exception as e:
    print(f"[WARN] skipping sharing strategy setup: {e}")

# ---------------------------
# util: schedule... (unchanged)
# ---------------------------
def heads_for_epoch(CFG, epoch: int):
    # ... (unchanged) ...
    if hasattr(CFG, "heads_schedule") and CFG.heads_schedule:
        ep_per = int(getattr(CFG, "epochs_per_phase", 1))
        idx = min(epoch // max(1, ep_per), len(CFG.heads_schedule) - 1)
        return list(CFG.heads_schedule[idx])
    if hasattr(CFG, "heads") and CFG.heads:
        return list(CFG.heads)
    return ["i", "v", "t", "ivt"]

def loss_weights(CFG):
    # ... (unchanged) ...
    base = {"i":1.0, "v":1.0, "t":1.0, "ivt":1.0}
    user = getattr(CFG, "loss_weights", None)
    if isinstance(user, dict):
        base.update(user)
    return base

def pick_ckpt_score(scores, mode="overall", weights=None):
    # ... (unchanged) ...
    if mode in ("tri", "ivt"):
        return float(scores.get("tri_mAP", -1.0))
    return float(scores.get("overall", -1.0))

# ---------------------------
# score formatting helper (unchanged)
# ---------------------------
def format_score(score, multiplier=100.0, default_val='-'):
    # ... (unchanged) ...
    if isinstance(score, (float, int)):
        return f"{score * multiplier:.1f}"
    return default_val

# ---------------------------

# ---------------------------
def train_one_epoch(train_loader, model, CFG, criterion, optimizer, scaler, scheduler, epoch):
    # ... (unchanged, no change) ...
    model.train()
    accum = max(1, int(getattr(CFG, "gradient_accumulation_steps", 1)))
    optimizer.zero_grad(set_to_none=True)
    lw = loss_weights(CFG)
    active = heads_for_epoch(CFG, epoch)
    running = 0.0
    pbar = tqdm(train_loader, total=len(train_loader), desc=f"Train {epoch:02d}", leave=False)
    for step, batch in enumerate(pbar):
        if len(batch) == 3:
            images, labels_tuple, counts_tuple = batch
        else:
            images, labels_tuple = batch[:2]
            
        ivt, inst, verb, target, *_ = labels_tuple
        images      = images.to(CFG.device, non_blocking=True)
        inst_gt   = inst.to(CFG.device)
        verb_gt   = verb.to(CFG.device)
        target_gt = target.to(CFG.device)
        ivt_gt    = ivt.to(CFG.device)
        
        with amp.autocast():
            out = model(images)
            loss = 0.0
            if "i" in active:   loss += lw["i"]   * criterion(out["i"],   inst_gt)
            if "t" in active:   loss += lw["t"]   * criterion(out["t"],   target_gt)
            if "v" in active:   loss += lw["v"]   * criterion(out["v"],   verb_gt)
            if "ivt" in active: loss += lw["ivt"] * criterion(out["ivt"], ivt_gt)
        loss = loss / accum
        scaler.scale(loss).backward()
        if (step + 1) % accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        running += loss.item() * accum
        pbar.set_postfix(loss=f"{(running/(step+1)):.4f}")
    if scheduler is not None:
        scheduler.step()
    return running / max(1, len(train_loader))

@torch.no_grad()
# [★ key change 1] added description arg
def validate_one_epoch(valid_loader, model, CFG, criterion, epoch, description=""):
    model.eval()
    active = heads_for_epoch(CFG, epoch)
    preds_i, preds_v, preds_t, preds_ivt = [], [], [], []
    running = 0.0
    
    # [★ key change 2] use given description
    if not description:
        description = f"Valid {epoch:02d}" # default value
    pbar = tqdm(valid_loader, total=len(valid_loader), desc=description, leave=False) 
    
    for step, batch in enumerate(pbar):
        if len(batch) == 3:
            images, labels_tuple, counts_tuple = batch
        else:
            images, labels_tuple = batch[:2]
        ivt, inst, verb, target, *_ = labels_tuple
        images      = images.to(CFG.device, non_blocking=True)
        inst_gt   = inst.to(CFG.device)
        verb_gt   = verb.to(CFG.device)
        target_gt = target.to(CFG.device)
        ivt_gt    = ivt.to(CFG.device)
        logits = model(images)
        loss = 0.0
        if "i" in active:   loss += criterion(logits["i"], inst_gt)
        if "v" in active:   loss += criterion(logits["v"], verb_gt)
        if "t" in active:   loss += criterion(logits["t"], target_gt)
        if "ivt" in active: loss += criterion(logits["ivt"], ivt_gt)
        running += loss.item()
        if "i" in active:   preds_i.append(torch.sigmoid(logits["i"]).cpu().numpy())
        if "v" in active:   preds_v.append(torch.sigmoid(logits["v"]).cpu().numpy())
        if "t" in active:   preds_t.append(torch.sigmoid(logits["t"]).cpu().numpy())
        if "ivt" in active: preds_ivt.append(torch.sigmoid(logits["ivt"]).cpu().numpy())
        avg = running / (step + 1)
        pbar.set_postfix(loss=f"{avg:.4f}")
    preds = {}
    if preds_i:   preds["i"]   = np.concatenate(preds_i,   axis=0)
    if preds_v:   preds["v"]   = np.concatenate(preds_v,   axis=0)
    if preds_t:   preds["t"]   = np.concatenate(preds_t,   axis=0)
    if preds_ivt: preds["ivt"] = np.concatenate(preds_ivt, axis=0)
    return running / max(1, len(valid_loader)), preds

@torch.no_grad()
def compute_maps(CFG, valid_folds, preds_dict, active_heads):
    # ... (unchanged, no change) ...
    from utils import (
        per_epoch_ivtmetrics_inst,
        per_epoch_ivtmetrics_verb,
        per_epoch_ivtmetrics_target,
        per_epoch_ivtmetrics_all
    )
    scores = {}
    df_meta_cols = [c for c in ['folder','frame','video','image_path','image_id','fold'] if c in valid_folds.columns]
    n_inst   = getattr(CFG, "n_inst", 6)
    n_verb   = getattr(CFG, "n_verb", 10)
    n_target = getattr(CFG, "n_target", 15)
    n_triple_csv = getattr(CFG, "n_triple", 170)
    
    if "i" in active_heads and "i" in preds_dict:
        gt_cols = [f'inst{c}' for c in range(n_inst)]
        n_items = len(gt_cols)
        assert preds_dict["i"].shape[1] == n_items, f"Inst Mismatch: Pred={preds_dict['i'].shape[1]}, CFG={n_items}"
        df = valid_folds[df_meta_cols + gt_cols].copy()
        for c_idx in range(n_items):
            df[str(c_idx)] = preds_dict["i"][:, c_idx]
        scores["i_mAP"] = float(per_epoch_ivtmetrics_inst(df, CFG))
    
    if "v" in active_heads and "v" in preds_dict:
        gt_cols = [f'v{c}' for c in range(n_verb)]
        n_items = len(gt_cols)
        assert preds_dict["v"].shape[1] == n_items, f"Verb Mismatch: Pred={preds_dict['v'].shape[1]}, CFG={n_items}"
        df = valid_folds[df_meta_cols + gt_cols].copy()
        for c_idx in range(n_items):
            df[str(c_idx)] = preds_dict["v"][:, c_idx]
        scores["v_mAP"] = float(per_epoch_ivtmetrics_verb(df, CFG))
    
    if "t" in active_heads and "t" in preds_dict:
        gt_cols = [f't{c}' for c in range(n_target)]
        n_items = len(gt_cols)
        assert preds_dict["t"].shape[1] == n_items, f"Target Mismatch: Pred={preds_dict['t'].shape[1]}, CFG={n_items}"
        df = valid_folds[df_meta_cols + gt_cols].copy()
        for c_idx in range(n_items):
            df[str(c_idx)] = preds_dict["t"][:, c_idx]
        scores["t_mAP"] = float(per_epoch_ivtmetrics_target(df, CFG))
    
    if "ivt" in active_heads and "ivt" in preds_dict:
        gt_cols = [f'tri{c}' for c in range(n_triple_csv)]
        n_items = len(gt_cols)
        assert preds_dict["ivt"].shape[1] == n_items, f"Triplet Mismatch: Pred={preds_dict['ivt'].shape[1]}, GT_Cols={n_items}"
        df = valid_folds[df_meta_cols + gt_cols].copy()
        for c_idx in range(n_items): 
            df[str(c_idx)] = preds_dict["ivt"][:, c_idx]
        tri_mAP, i_mAP, v_mAP, t_mAP = per_epoch_ivtmetrics_all(df, CFG)
        scores["tri_mAP"] = float(tri_mAP)
        if "i_mAP" not in scores: scores["i_mAP"] = float(i_mAP)
        if "v_mAP" not in scores: scores["v_mAP"] = float(v_mAP)
        if "t_mAP" not in scores: scores["t_mAP"] = float(t_mAP)
    
    present = []
    if "i"   in active_heads and "i_mAP"   in scores: present.append(scores["i_mAP"])
    if "v"   in active_heads and "v_mAP"   in scores: present.append(scores["v_mAP"])
    if "t"   in active_heads and "t_mAP"   in scores: present.append(scores["t_mAP"])
    if "ivt" in active_heads and "tri_mAP" in scores: present.append(scores["tri_mAP"])
    scores["overall"] = float(np.mean(present)) if present else 0.0
    return scores

def _video_to_num(s: str):
    # ... (unchanged) ...
    s = str(s)
    if s.isdigit():
        return int(s)
    m = re.findall(r"\d+", s)
    return int(m[0]) if m else -1

# ---------------------------
# fold attach util (unchanged)
# ---------------------------
def attach_folds_by_image_or_video(fused_full, folds):
    # ... (unchanged) ...
    need_cols = {"image_id","video","fold"}
    assert need_cols.issubset(set(folds.columns)), f"[attach_folds] folds requires {need_cols}."
    fused_full = fused_full.copy()
    if 'fold' in fused_full.columns:
        fused_full = fused_full.drop(columns=['fold'])
    base = folds[['image_id','video','fold']].drop_duplicates().copy()
    fused_full['image_id'] = fused_full['image_id'].astype(str)
    base['image_id']       = base['image_id'].astype(str)
    fused_full['video'] = fused_full['video'].astype(str)
    base['video']       = base['video'].astype(str)
    fused_full = fused_full.merge(base[['image_id','fold']], on='image_id', how='left')
    if fused_full['fold'].isna().any():
        missing_rows = fused_full['fold'].isna()
        vmap = base[['video','fold']].drop_duplicates().set_index('video')
        video_folds = fused_full.loc[missing_rows, 'video'].map(vmap['fold'])
        fused_full.loc[missing_rows, 'fold'] = video_folds.values
    fused_full['fold'] = fused_full['fold'].fillna(-1).astype(int)
    return fused_full

# ---------------------------

# ---------------------------
def train_cross_val(CFG):
    start_time = time.time()
    seed_torch(seed=CFG.seed)
    out_dir  = CFG.output_dir
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    folds = get_folds(CFG)

    # --- [★ key change 1] load single Train DF ---
    try:
        print(f"[Data] Loading train data from: {CFG.train_csv_path}")
        train_df_all = pd.read_csv(to_absolute_path(CFG.train_csv_path))
    except FileNotFoundError as e:
        print(f"[ERROR] Train CSV file not found: {e.fname}")
        raise e
    except AttributeError as e:
        print(f"[ERROR] 'train_csv_path' missing in config.")
        raise e
        
    print("[Data] Attaching fold information to Train DF...")
    train_df_all = attach_folds_by_image_or_video(train_df_all, folds)
    print(f"[Data] Train DF Fold (post-attach): {dict(train_df_all['fold'].value_counts().sort_index())}")


    # --- [★ key change 2] load multi Validation DFs ---
    val_scenarios = getattr(CFG, "val_strategies", []) 
    val_csv_template = getattr(CFG, "val_csv_template", "cholecT50_{strategy}.csv")
    data_base_dir = getattr(CFG, "data_base_dir", ".")
    
    val_dfs_all = {} 
    
    if not val_scenarios:
        print("[WARN] 'val_strategies' is empty in config; skipping validation.")
    
    for scenario in val_scenarios:
        csv_file = val_csv_template.format(strategy=scenario)
        csv_path = os.path.join(data_base_dir, csv_file)
        try:
            print(f"[Data] Loading val data for '{scenario}' from: {csv_path}")
            df = pd.read_csv(to_absolute_path(csv_path))
            print(f"[Data] Attaching fold information to Val DF ({scenario})...")
            df_folded = attach_folds_by_image_or_video(df, folds)
            val_dfs_all[scenario] = df_folded
            print(f"[Data] Val DF Fold ({scenario}) (post-attach): {dict(df_folded['fold'].value_counts().sort_index())}")
        except FileNotFoundError as e:
            print(f"[ERROR] '{scenario}' CSV file not found: {e.fname}")
            print(f"       skipping this scenario.")
        except Exception as e:
            print(f"[ERROR] '{scenario}' unknown error during load: {e}")

    if not val_dfs_all:
         print("[WARN] No valid validation CSV loaded; training will proceed without validation.")
    
    
    all_scores_summary = {
        scenario: {
            "overall": [], "i": [], "v": [], "t": [], "ivt": []
        } for scenario in val_dfs_all.keys()
    }


    for fold in range(CFG.start_n_fold, CFG.n_fold):
        if fold not in CFG.trn_fold:
            continue

        
        # --- [★ key change 4] 1Trainset + NValidsets prepared ---
        
        
        train_folds = train_df_all[train_df_all["fold"] != fold].reset_index(drop=True)

        if len(train_folds) == 0:
            print(f"[WARN] fold {fold} train set is empty. Skipping this fold.")
            continue
            
        train_dataset = TrainDataset(train_folds, CFG, 
                                     transform=get_transforms(data="train", CFG=CFG), 
                                     fold=fold, 
                                     mode="normal")
        
        train_loader  = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True,
                                   num_workers=CFG.nworkers, pin_memory=True, drop_last=True)
        
        
        
        valid_loaders = {}   
        valid_datasets = {}  
        
        for scenario, df_all in val_dfs_all.items():
            valid_folds_sc = df_all[df_all["fold"] == fold].reset_index(drop=True)

            if len(valid_folds_sc) == 0:
                print(f"  [WARN] fold {fold}, scenario '{scenario}' validation set is empty. Skipping.")
                continue

            valid_dataset_sc = TrainDataset(valid_folds_sc, CFG, 
                                            transform=get_transforms(data="valid", CFG=CFG), 
                                            fold=fold, 
                                            mode=getattr(CFG, "val_data_mode", "intersection"))
            
            valid_loader_sc = DataLoader(valid_dataset_sc, batch_size=CFG.valid_batch_size, shuffle=False,
                                         num_workers=CFG.nworkers, pin_memory=False, drop_last=False)
            
            valid_loaders[scenario] = valid_loader_sc
            valid_datasets[scenario] = valid_dataset_sc 
            
        if not valid_loaders:
             print(f"[WARN] fold {fold} has no valid validation loaders. This fold will only train.")
             
        active_scenarios = list(valid_loaders.keys()) 

        model = SwinIVTModel(CFG, CFG.model_name, pretrained=CFG.pretrained).to(CFG.device)
        optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=getattr(CFG, "T_0", CFG.epochs + 1), T_mult=1000, eta_min=CFG.min_lr
        ) if getattr(CFG, "use_scheduler", True) else None
        criterion = nn.BCEWithLogitsLoss(reduction="mean").to(CFG.device)
        scaler = amp.GradScaler()

        # --- [★ key change 5] track per-scenario best score and path ---
        best_scores = {scenario: -1.0 for scenario in active_scenarios}
        best_paths = {
            scenario: os.path.join(ckpt_dir, f"fold{fold}_{CFG.model_name[:8]}_{CFG.exp}_best_{scenario}.pth")
            for scenario in active_scenarios
        }
        best_epoch_scores = {scenario: {} for scenario in active_scenarios} 
        
        print("start training")

        for epoch in range(CFG.epochs):
            active_heads = heads_for_epoch(CFG, epoch)
            print(f"\n[Epoch {epoch:02d}] active heads = {active_heads}")

            t0 = time.time()
            
            
            tr_loss = train_one_epoch(train_loader, model, CFG, criterion, optimizer, scaler, scheduler, epoch)
            
            
            # --- [★ key change 6] Niterate Validation scenarios ---
            print(f"--- Epoch {epoch:02d} Validation ---")
            all_scenario_scores_this_epoch = {} 
            
            for scenario in active_scenarios:
                va_loader_sc = valid_loaders[scenario]
                va_dataset_sc = valid_datasets[scenario] 
                
                
                
                
                desc_str = f"Valid {epoch:02d} [{scenario}]"
                va_loss, preds = validate_one_epoch(
                    va_loader_sc, model, CFG, criterion, epoch, description=desc_str
                )
                
                
                scores = compute_maps(CFG, va_dataset_sc.df, preds, active_heads)
                all_scenario_scores_this_epoch[scenario] = (va_loss, scores)


            # --- [★ key change 7] Epoch summary and per-scenario checkpoint save ---
            took = (time.time() - t0) / 60.0
            print(f"--- Epoch {epoch:02d} Summary (TrainLoss {tr_loss:.4f} | time {took:.2f}m) ---")
            
            for scenario in active_scenarios:
                if scenario not in all_scenario_scores_this_epoch:
                    continue
                    
                va_loss, scores = all_scenario_scores_this_epoch[scenario]
                
                msg = (f"  [{scenario:12}] Valid {va_loss:.4f} | " 
                       f"overall {scores['overall'] * 100:.1f} | ")
                extras = []
                if "i_mAP"   in scores: extras.append(f"i {scores['i_mAP'] * 100:.1f}")
                if "v_mAP"   in scores: extras.append(f"v {scores['v_mAP'] * 100:.1f}")
                if "t_mAP"   in scores: extras.append(f"t {scores['t_mAP'] * 100:.1f}")
                if "tri_mAP" in scores: extras.append(f"ivt {scores['tri_mAP'] * 100:.1f}")
                if extras: msg += " / ".join(extras)
                print(msg)

                ckpt_mode = getattr(CFG, "ckpt_metric", "tri")
                ckpt_weights = getattr(CFG, "ckpt_weights", None)
                metric_val = pick_ckpt_score(scores, ckpt_mode, ckpt_weights)
                
                if metric_val > best_scores[scenario]:
                    best_scores[scenario] = metric_val
                    best_epoch_scores[scenario] = scores.copy() 
                    
                    save_path = best_paths[scenario] 
                    os.makedirs(os.path.dirname(save_path), exist_ok=True) 
                    torch.save({"model": model.state_dict()}, save_path)
                    print(f"    [CKPT] saved -> {save_path} (metric={ckpt_mode}, value={metric_val * 100:.1f})")

        
        # --- [★ key change 8] report at fold end (Nscenarios) ---
        print(f"\n--- [Fold {fold} Best Scores Summary] ---")
        
        for scenario in active_scenarios:
            final_scores = best_epoch_scores.get(scenario, {}) 
            
            if not final_scores:
                 print(f"  [{scenario:12}] No best score recorded.")
                 continue

            print(f"  [{scenario:12}] best overall: {format_score(final_scores.get('overall'))} "
                  f"(i:{format_score(final_scores.get('i_mAP'))}, v:{format_score(final_scores.get('v_mAP'))}, "
                  f"t:{format_score(final_scores.get('t_mAP'))}, ivt:{format_score(final_scores.get('tri_mAP'))})")
            
            if 'overall' in final_scores:
                all_scores_summary[scenario]["overall"].append(final_scores['overall'])
            if 'i_mAP' in final_scores:
                all_scores_summary[scenario]["i"].append(final_scores['i_mAP'])
            if 'v_mAP' in final_scores:
                all_scores_summary[scenario]["v"].append(final_scores['v_mAP'])
            if 't_mAP' in final_scores:
                all_scores_summary[scenario]["t"].append(final_scores['t_mAP'])
            if 'tri_mAP' in final_scores:
                all_scores_summary[scenario]["ivt"].append(final_scores['tri_mAP'])
            
        del model, train_loader, valid_loaders, valid_datasets 
        torch.cuda.empty_cache()

    
    # --- [★ key change 9] final summary print (per-scenario detail) ---
    print("\n" + "="*60)
    print(f"🏁 Cross-Validation Summary")
    print("="*60)
    
    for scenario, scores_dict in all_scores_summary.items():
        
        if not scores_dict["overall"]:
            print(f"\n--- Scenario: {scenario} ---")
            print("  (No scores recorded for this scenario)")
            continue

        print(f"\n--- Scenario: {scenario} (n={len(scores_dict['overall'])} folds) ---")
        
        mean_overall = np.mean(scores_dict["overall"])
        std_overall = np.std(scores_dict["overall"])
        print(f"    - Mean Overall mAP : {mean_overall * 100:.1f}")
        print(f"    - Std Dev Overall mAP: {std_overall * 100:.1f}")
        
        if scores_dict["i"]: 
            mean_i = np.mean(scores_dict["i"])
            std_i = np.std(scores_dict["i"])
            print(f"    - Mean I mAP         : {mean_i * 100:.1f}")
            print(f"    - Std Dev I mAP      : {std_i * 100:.1f}")
            
        if scores_dict["v"]:
            mean_v = np.mean(scores_dict["v"])
            std_v = np.std(scores_dict["v"])
            print(f"    - Mean V mAP         : {mean_v * 100:.1f}")
            print(f"    - Std Dev V mAP      : {std_v * 100:.1f}")
            
        if scores_dict["t"]:
            mean_t = np.mean(scores_dict["t"])
            std_t = np.std(scores_dict["t"])
            print(f"    - Mean T mAP         : {mean_t * 100:.1f}")
            print(f"    - Std Dev T mAP      : {std_t * 100:.1f}")
            
        if scores_dict["ivt"]:
            mean_ivt = np.mean(scores_dict["ivt"])
            std_ivt = np.std(scores_dict["ivt"])
            print(f"    - Mean IVT mAP       : {mean_ivt * 100:.1f}")
            print(f"    - Std Dev IVT mAP    : {std_ivt * 100:.1f}")

    print("="*60 + "\n")
        
    print(f"Training time: {(time.time() - start_time)/60:.2f} minutes")

# main entry
if __name__ == "__main__":
    
    print("train.py: This file is designed to be imported by a main script.")
    print("`train_cross_val(CFG)` Call the function to run.")