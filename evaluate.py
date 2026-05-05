#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from preprocess import get_folds
from augmentation import get_transforms
from models import SwinIVTModel
from dataset import TrainDataset

# ---------------------------------------------------------
# Global knobs
# ---------------------------------------------------------
TRAIN_SCENARIO_NAME = "Soft"
OUTPUT_DIR_NAME = "[EVAL]logs_CoAP_HardAP_MAE"
CONFIG_NAME = "config"
TABLE_LOG_NAME_PREFIX = f"Eval_CoAP_HardAP_MAE_{TRAIN_SCENARIO_NAME}"

THR_POS = 1e-4  # for MAE positive-only mask

# consensus-aware thresholds (CoAP)
DEFAULT_CA_TAUS = [0.3, 0.6, 1.0]  # override with cfg.consensus_mAP_taus
CA_EPS = 0.0  # optional safety margin for binarization

# conventional HardAP threshold (single-threshold binarization)
DEFAULT_HARD_MAP_THR = 0.5  # override with cfg.hard_map_thr


# =========================================================
# 0) Simple table logging
# =========================================================
_TABLE_LOG_FH = None

def setup_table_log(outdir_name, prefix="tables_only"):
    global _TABLE_LOG_FH
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(PROJECT_ROOT, outdir_name)
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{prefix}.log")
    _TABLE_LOG_FH = open(path, "w", encoding="utf-8")
    print(f"[TABLE-LOG] -> {path}")
    return path

def close_table_log():
    global _TABLE_LOG_FH
    if _TABLE_LOG_FH is not None:
        try:
            _TABLE_LOG_FH.flush()
            _TABLE_LOG_FH.close()
        except Exception:
            pass
        _TABLE_LOG_FH = None

def tprint(*args, **kwargs):
    global _TABLE_LOG_FH
    print(*args, **kwargs)
    if _TABLE_LOG_FH is None:
        return
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    msg = sep.join(str(a) for a in args) + end
    _TABLE_LOG_FH.write(msg)
    _TABLE_LOG_FH.flush()

def _safe_mean(x, default=0.0):
    x = list(x)
    return float(np.mean(x)) if len(x) > 0 else float(default)

def _format_mean_std(mean_val, std_val, scale=100.0, nd=1):
    if mean_val is None or std_val is None or pd.isna(mean_val) or pd.isna(std_val):
        return "-"
    return f"{mean_val * scale:.{nd}f} ± {std_val * scale:.{nd}f}"


# =========================================================
# 1) Column helpers
# =========================================================
def get_gt_columns(df, prefix):
    pattern = re.compile(rf"^{prefix}\d+$")
    cols = [c for c in df.columns if pattern.match(c)]
    cols.sort(key=lambda x: int(x.replace(prefix, "")))
    return cols

def binarize_gt_for_hardap(df, prefixes, thr=0.5, eps=0.0):
    dfb = df.copy()
    thr_eff = float(thr) - float(eps)
    for pref in prefixes:
        cols = [c for c in dfb.columns if re.match(rf"^{pref}\d+$", c)]
        if not cols:
            continue
        dfb[cols] = (dfb[cols].values >= thr_eff).astype(np.int32)
    return dfb


# =========================================================
# 2) MAE (positive-only)
# =========================================================
def mae_positive_only(preds, targets, thr=THR_POS):
    if preds is None or targets is None:
        return 0.0
    mask = targets > thr
    if np.sum(mask) == 0:
        return 0.0
    return float(np.mean(np.abs(preds[mask] - targets[mask])))


# =========================================================
# 3) Projection matrices (IVT -> I/V/T)
# =========================================================
def infer_projection_matrices(df_soft, tri_cols, inst_cols, v_cols, t_cols, device="cpu"):
    """
    Build projection matrices:
      P_I: [n_tri, n_inst]
      P_V: [n_tri, n_verb]
      P_T: [n_tri, n_target]
    by selecting the most frequent component label among samples where each triplet is present.
    """
    tprint(" 🤖 [Auto-Mapping] Inferring IVT -> I/V/T projection matrices from GT co-occurrence...")

    gt_tri  = df_soft[tri_cols].values
    gt_inst = df_soft[inst_cols].values if len(inst_cols) else None
    gt_v    = df_soft[v_cols].values if len(v_cols) else None
    gt_t    = df_soft[t_cols].values if len(t_cols) else None

    n_tri = len(tri_cols)
    P_I = torch.zeros((n_tri, len(inst_cols)), device=device) if gt_inst is not None else None
    P_V = torch.zeros((n_tri, len(v_cols)),    device=device) if gt_v    is not None else None
    P_T = torch.zeros((n_tri, len(t_cols)),    device=device) if gt_t    is not None else None

    resolved = {"I": 0, "V": 0, "T": 0}

    for j in range(n_tri):
        mask = gt_tri[:, j] > 0.1
        if np.sum(mask) == 0:
            continue

        if P_I is not None:
            idx_i = int(np.argmax(np.sum(gt_inst[mask], axis=0)))
            P_I[j, idx_i] = 1.0
            resolved["I"] += 1
        if P_V is not None:
            idx_v = int(np.argmax(np.sum(gt_v[mask], axis=0)))
            P_V[j, idx_v] = 1.0
            resolved["V"] += 1
        if P_T is not None:
            idx_t = int(np.argmax(np.sum(gt_t[mask], axis=0)))
            P_T[j, idx_t] = 1.0
            resolved["T"] += 1

    tprint(f"    ✅ mapped triplets -> I: {resolved['I']}/{n_tri}, V: {resolved['V']}/{n_tri}, T: {resolved['T']}/{n_tri}")
    return {"I": P_I, "V": P_V, "T": P_T}


# =========================================================
# 4) HardAP(mAP) via utils_full
# =========================================================
def _import_utils_full():
    try:
        from utils_full import (
            per_epoch_ivtmetrics_inst,
            per_epoch_ivtmetrics_verb,
            per_epoch_ivtmetrics_target,
            per_epoch_ivtmetrics_all,
        )
        return per_epoch_ivtmetrics_inst, per_epoch_ivtmetrics_verb, per_epoch_ivtmetrics_target, per_epoch_ivtmetrics_all
    except ImportError:
        tprint("[ERROR] utils_full.py not found. Cannot compute HardAP(mAP)/CA-mAP.")
        return None

def hardap_ivt(CFG, df_hard, preds_ivt):
    funcs = _import_utils_full()
    if funcs is None:
        return 0.0
    _, _, _, per_epoch_ivtmetrics_all = funcs

    df_for_map = df_hard.reset_index(drop=True)
    preds_df = pd.DataFrame(preds_ivt, columns=[str(c) for c in range(CFG.n_triple)])
    out = per_epoch_ivtmetrics_all(pd.concat([df_for_map, preds_df], axis=1), CFG)
    tri_mAP = float(out[0]) if isinstance(out, (list, tuple)) else float(out)
    return tri_mAP

def hardap_component(CFG, df_hard, preds_comp, comp_key):
    funcs = _import_utils_full()
    if funcs is None:
        return 0.0
    per_epoch_ivtmetrics_inst, per_epoch_ivtmetrics_verb, per_epoch_ivtmetrics_target, _ = funcs

    df_for_map = df_hard.reset_index(drop=True)
    if comp_key == "I":
        preds_df = pd.DataFrame(preds_comp, columns=[str(c) for c in range(CFG.n_inst)])
        return float(per_epoch_ivtmetrics_inst(pd.concat([df_for_map, preds_df], axis=1), CFG))
    if comp_key == "V":
        preds_df = pd.DataFrame(preds_comp, columns=[str(c) for c in range(CFG.n_verb)])
        return float(per_epoch_ivtmetrics_verb(pd.concat([df_for_map, preds_df], axis=1), CFG))
    if comp_key == "T":
        preds_df = pd.DataFrame(preds_comp, columns=[str(c) for c in range(CFG.n_target)])
        return float(per_epoch_ivtmetrics_target(pd.concat([df_for_map, preds_df], axis=1), CFG))
    raise ValueError(f"Unknown comp_key={comp_key}")


# =========================================================
# 5) Consensus-aware mAP (avg + per-τ)
# =========================================================
def _empty_ca_dict(taus):
    return {"avg": 0.0, "per_tau": {float(t): 0.0 for t in taus}}

def consensus_aware_map_ivt(CFG, df_soft, preds_ivt, taus, eps=0.0):
    if preds_ivt is None:
        return _empty_ca_dict(taus)

    scores = []
    per_tau = {}
    for tau in taus:
        df_hard = binarize_gt_for_hardap(df_soft, prefixes=("tri",), thr=float(tau), eps=float(eps))
        s = hardap_ivt(CFG, df_hard, preds_ivt)
        per_tau[float(tau)] = s
        scores.append(s)
    return {"avg": float(np.mean(scores)) if scores else 0.0, "per_tau": per_tau}

def consensus_aware_map_component(CFG, df_soft, preds_comp, comp_key, taus, eps=0.0):
    if preds_comp is None:
        return _empty_ca_dict(taus)

    pref = {"I": "inst", "V": "v", "T": "t"}[comp_key]
    scores = []
    per_tau = {}
    for tau in taus:
        df_hard = binarize_gt_for_hardap(df_soft, prefixes=(pref,), thr=float(tau), eps=float(eps))
        s = hardap_component(CFG, df_hard, preds_comp, comp_key)
        per_tau[float(tau)] = s
        scores.append(s)
    return {"avg": float(np.mean(scores)) if scores else 0.0, "per_tau": per_tau}


# =========================================================
# 6) Inference (ONLY need ivt head)
# =========================================================
@torch.no_grad()
def inference_fn(model, dataloader, device):
    model.eval()
    preds_chunks = []

    for batch in tqdm(dataloader, desc="Inference", leave=False):
        if isinstance(batch, dict):
            imgs = batch["image"]
        elif isinstance(batch, (list, tuple)):
            imgs = batch[0]
        else:
            imgs = batch
        imgs = imgs.to(device, non_blocking=True)

        logits = model(imgs)
        if "ivt" not in logits:
            raise KeyError("Model output does not contain 'ivt' head.")
        preds_chunks.append(torch.sigmoid(logits["ivt"]).cpu().numpy())

    return np.concatenate(preds_chunks, axis=0) if len(preds_chunks) else None


# =========================================================
# 7) Folding helper
# =========================================================
def attach_folds_by_image_or_video(fused_full, folds):
    need_cols = {"image_id", "video", "fold"}
    assert need_cols.issubset(set(folds.columns)), f"Folds must have {need_cols}"
    fused_full = fused_full.copy()
    if "fold" in fused_full.columns:
        fused_full = fused_full.drop(columns=["fold"])

    base = folds[["image_id", "video", "fold"]].drop_duplicates().copy()
    fused_full["image_id"] = fused_full["image_id"].astype(str)
    base["image_id"] = base["image_id"].astype(str)
    fused_full["video"] = fused_full["video"].astype(str)
    base["video"] = base["video"].astype(str)

    fused_full = fused_full.merge(base[["image_id", "fold"]], on="image_id", how="left")
    if fused_full["fold"].isna().any():
        missing = fused_full["fold"].isna()
        vmap = base[["video", "fold"]].drop_duplicates().set_index("video")
        fused_full.loc[missing, "fold"] = fused_full.loc[missing, "video"].map(vmap["fold"]).values

    fused_full["fold"] = fused_full["fold"].fillna(-1).astype(int)
    return fused_full


# =========================================================
# 8) Printing
# =========================================================
def print_summary_table(df_res):
    """
    Three metrics × {IVT, I_proj, V_proj, T_proj}:
      - CoAP   (CAIVT_avg / CAI_proj_avg / CAV_proj_avg / CAT_proj_avg)
      - HardAP (HardAP_IVT / HardAP_I_proj / HardAP_V_proj / HardAP_T_proj)
      - MAE    (MAE_IVT   / MAE_I_proj   / MAE_V_proj   / MAE_T_proj)  -- printed as %
    """
    mean_res = df_res.mean(numeric_only=True)
    std_res  = df_res.std(numeric_only=True)

    tprint("\n" + "="*120)
    tprint("📌 Summary (Mean ± Std over folds)")
    tprint("="*120)

    # CoAP (↑)
    tprint("\n[CoAP (↑)]")
    tprint(f"  IVT (ivt-head):            {_format_mean_std(mean_res.get('CAIVT_avg'),    std_res.get('CAIVT_avg'),    scale=100.0, nd=1)}")
    tprint("  I/V/T (IVT-projection-based):")
    tprint(f"    I (ivt→I):               {_format_mean_std(mean_res.get('CAI_proj_avg'), std_res.get('CAI_proj_avg'), scale=100.0, nd=1)}")
    tprint(f"    V (ivt→V):               {_format_mean_std(mean_res.get('CAV_proj_avg'), std_res.get('CAV_proj_avg'), scale=100.0, nd=1)}")
    tprint(f"    T (ivt→T):               {_format_mean_std(mean_res.get('CAT_proj_avg'), std_res.get('CAT_proj_avg'), scale=100.0, nd=1)}")

    # HardAP (↑)
    tprint("\n[HardAP / mAP (↑)]")
    tprint(f"  IVT (ivt-head):            {_format_mean_std(mean_res.get('HardAP_IVT'),    std_res.get('HardAP_IVT'),    scale=100.0, nd=1)}")
    tprint("  I/V/T (IVT-projection-based):")
    tprint(f"    I (ivt→I):               {_format_mean_std(mean_res.get('HardAP_I_proj'), std_res.get('HardAP_I_proj'), scale=100.0, nd=1)}")
    tprint(f"    V (ivt→V):               {_format_mean_std(mean_res.get('HardAP_V_proj'), std_res.get('HardAP_V_proj'), scale=100.0, nd=1)}")
    tprint(f"    T (ivt→T):               {_format_mean_std(mean_res.get('HardAP_T_proj'), std_res.get('HardAP_T_proj'), scale=100.0, nd=1)}")

    # MAE (↓) pos-only, printed as %
    tprint("\n[MAE (↓)  pos-only, %]")
    def fmt_mae_pct(key):
        mv = mean_res.get(key, None)
        sv = std_res.get(key, None)
        if mv is None or sv is None or pd.isna(mv) or pd.isna(sv):
            return "-"
        return f"{mv*100.0:.1f} ± {sv*100.0:.1f}"

    tprint(f"  IVT (ivt-head):            {fmt_mae_pct('MAE_IVT')}")
    tprint("  I/V/T (IVT-projection-based):")
    tprint(f"    I (ivt→I):               {fmt_mae_pct('MAE_I_proj')}")
    tprint(f"    V (ivt→V):               {fmt_mae_pct('MAE_V_proj')}")
    tprint(f"    T (ivt→T):               {fmt_mae_pct('MAE_T_proj')}")

    tprint("="*120)




# =========================================================
# 9) Main evaluation loop
# =========================================================
def run_evaluation(CFG: DictConfig):
    tprint("="*120)
    tprint("🚀 Evaluation: CoAP + HardAP(mAP) + MAE  (IVT-head + IVT-projection for I/V/T)")
    tprint("="*120)

    ckpt_dir = os.path.join(to_absolute_path(CFG.output_dir), "checkpoints")
    folds = get_folds(CFG)
    if len(folds) == 0:
        tprint("[ERROR] folds empty")
        return

    device = torch.device(CFG.device)
    val_transform = get_transforms(data="valid", CFG=CFG)

    csv_path = os.path.join(to_absolute_path(CFG.data_base_dir), CFG.val_csv_template.format(strategy="Soft"))
    df_soft_all = pd.read_csv(csv_path)
    df_soft_all = attach_folds_by_image_or_video(df_soft_all, folds)

    tri_cols  = get_gt_columns(df_soft_all, "tri")
    inst_cols = get_gt_columns(df_soft_all, "inst")
    v_cols    = get_gt_columns(df_soft_all, "v")
    t_cols    = get_gt_columns(df_soft_all, "t")

    assert len(tri_cols)  > 0, "No tri\\d+ columns found in CSV."
    assert len(inst_cols) > 0, "No inst\\d+ columns found in CSV."
    assert len(v_cols)    > 0, "No v\\d+ columns found in CSV."
    assert len(t_cols)    > 0, "No t\\d+ columns found in CSV."

    proj = infer_projection_matrices(df_soft_all, tri_cols, inst_cols, v_cols, t_cols, device=device)

    CA_TAUS = list(getattr(CFG, "consensus_mAP_taus", DEFAULT_CA_TAUS))
    CA_EPS_ = float(getattr(CFG, "consensus_mAP_eps", CA_EPS))
    HARD_THR = float(getattr(CFG, "hard_map_thr", DEFAULT_HARD_MAP_THR))

    # paper tag mapping (index-based)
    _alias = {0: "0.33", 1: "0.66", 2: "1.00"}

    val_scenarios = CFG.val_strategies

    for scenario in val_scenarios:
        tprint(f"\n\n==================== Scenario: {scenario} ====================")
        fold_rows = []

        for fold in CFG.trn_fold:
            ckpt_path = os.path.join(ckpt_dir, f"fold{fold}_{CFG.model_name[:8]}_{CFG.exp}_best_{scenario}.pth")
            if not os.path.exists(ckpt_path):
                tprint(f"[CKPT] fold {fold}: MISSING -> {ckpt_path}")
                continue

            df_fold = df_soft_all[df_soft_all["fold"] == fold].reset_index(drop=True)
            if len(df_fold) == 0:
                continue

            val_dataset = TrainDataset(
                df_fold, CFG,
                transform=val_transform,
                mode=CFG.val_data_mode,
                inference=True
            )
            if len(val_dataset) == 0:
                continue

            df_eval = val_dataset.df.reset_index(drop=True)

            val_loader = DataLoader(
                val_dataset,
                batch_size=CFG.valid_batch_size,
                shuffle=False,
                num_workers=CFG.nworkers,
            )

            model = SwinIVTModel(CFG, CFG.model_name, pretrained=False).to(device)
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state["model"])

            # Only ivt preds
            p_ivt = inference_fn(model, val_loader, device)  # [N, n_triple]
            if p_ivt is None:
                tprint(f"[WARN] fold {fold}: empty predictions -> skip")
                del model, val_loader, val_dataset
                torch.cuda.empty_cache()
                continue

            # Targets
            y_tri  = df_eval[tri_cols].values.astype(np.float32)
            y_inst = df_eval[inst_cols].values.astype(np.float32)
            y_v    = df_eval[v_cols].values.astype(np.float32)
            y_t    = df_eval[t_cols].values.astype(np.float32)

            # Projection-based predictions
            p_tri_t = torch.from_numpy(p_ivt).float().to(device)
            p_i_proj = (p_tri_t @ proj["I"]).detach().cpu().numpy()
            p_v_proj = (p_tri_t @ proj["V"]).detach().cpu().numpy()
            p_t_proj = (p_tri_t @ proj["T"]).detach().cpu().numpy()

            # CoAP (consensus-aware mAP)
            ca_ivt    = consensus_aware_map_ivt(CFG, df_eval, p_ivt,    taus=CA_TAUS, eps=CA_EPS_)
            ca_i_proj = consensus_aware_map_component(CFG, df_eval, p_i_proj, "I", taus=CA_TAUS, eps=CA_EPS_)
            ca_v_proj = consensus_aware_map_component(CFG, df_eval, p_v_proj, "V", taus=CA_TAUS, eps=CA_EPS_)
            ca_t_proj = consensus_aware_map_component(CFG, df_eval, p_t_proj, "T", taus=CA_TAUS, eps=CA_EPS_)

            # HardAP (conventional mAP at single threshold)
            df_hard_tri  = binarize_gt_for_hardap(df_eval, prefixes=("tri",), thr=HARD_THR, eps=CA_EPS_)
            df_hard_inst = binarize_gt_for_hardap(df_eval, prefixes=("inst",), thr=HARD_THR, eps=CA_EPS_)
            df_hard_v    = binarize_gt_for_hardap(df_eval, prefixes=("v",),    thr=HARD_THR, eps=CA_EPS_)
            df_hard_t    = binarize_gt_for_hardap(df_eval, prefixes=("t",),    thr=HARD_THR, eps=CA_EPS_)
            hardap_ivt_score    = hardap_ivt(CFG, df_hard_tri, p_ivt)
            hardap_i_proj_score = hardap_component(CFG, df_hard_inst, p_i_proj, "I")
            hardap_v_proj_score = hardap_component(CFG, df_hard_v,    p_v_proj, "V")
            hardap_t_proj_score = hardap_component(CFG, df_hard_t,    p_t_proj, "T")

            row = {
                "fold": fold,

                # CoAP avg
                "CAIVT_avg": ca_ivt["avg"],
                "CAI_proj_avg": ca_i_proj["avg"],
                "CAV_proj_avg": ca_v_proj["avg"],
                "CAT_proj_avg": ca_t_proj["avg"],

                # HardAP (conventional mAP)
                "HardAP_IVT":    hardap_ivt_score,
                "HardAP_I_proj": hardap_i_proj_score,
                "HardAP_V_proj": hardap_v_proj_score,
                "HardAP_T_proj": hardap_t_proj_score,

                # MAE pos-only (raw 0~1, printing will convert to %)
                "MAE_IVT": mae_positive_only(p_ivt,    y_tri,  thr=THR_POS),
                "MAE_I_proj": mae_positive_only(p_i_proj, y_inst, thr=THR_POS),
                "MAE_V_proj": mae_positive_only(p_v_proj, y_v,    thr=THR_POS),
                "MAE_T_proj": mae_positive_only(p_t_proj, y_t,    thr=THR_POS),
            }

            # CA per-tau (paper tags)
            for idx, tau in enumerate(CA_TAUS[:3]):
                tag = _alias.get(idx, f"{float(tau):.2f}")
                tau_f = float(tau)

                row[f"CAIVT@{tag}"]     = float(ca_ivt["per_tau"].get(tau_f, 0.0))
                row[f"CAI_proj@{tag}"]  = float(ca_i_proj["per_tau"].get(tau_f, 0.0))
                row[f"CAV_proj@{tag}"]  = float(ca_v_proj["per_tau"].get(tau_f, 0.0))
                row[f"CAT_proj@{tag}"]  = float(ca_t_proj["per_tau"].get(tau_f, 0.0))

            fold_rows.append(row)

            del model, val_loader, val_dataset
            torch.cuda.empty_cache()

        if len(fold_rows) == 0:
            tprint("[WARN] no folds evaluated for this scenario")
            continue

        df_res = pd.DataFrame(fold_rows)

        print_summary_table(df_res)



@hydra.main(config_path=".", config_name=CONFIG_NAME, version_base=None)
def main(CFG: DictConfig):
    start = time.time()
    log_path = None
    try:
        log_path = setup_table_log(outdir_name=OUTPUT_DIR_NAME, prefix=TABLE_LOG_NAME_PREFIX)
        run_evaluation(CFG)
        tprint(f"\nTotal Evaluation Time: {(time.time() - start) / 60:.2f} minutes")
    finally:
        close_table_log()
        if log_path is not None:
            print(f"[TABLE-LOG] Done. Saved: {log_path}")

if __name__ == "__main__":
    main()