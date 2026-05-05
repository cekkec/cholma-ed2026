# helper functions
import os
import random
import numpy as np
import torch
# import ivtmetrics
from ivtmetrics.recognition import Recognition
import pandas as pd 
from tqdm import tqdm 


def seed_torch(seed=42):
    """
    Seed various random number generators to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


### Table for printing results
# (table header/line definitions unchanged)
header = f"""
 Epoch | {"Loss":^6} | {"Val Loss":^7} | {"mAP":^7} | {"CmAP":^7} | {"Time, m":^6}
"""
header_v2 = f"""
 Epoch | {"Loss":^6} | {"Val Loss":^7} | {"Train_mAP":^7} | {"CmAP":^7} | {"Time, m":^6}
"""
header_split = f"""
 Epoch | {"Loss":^6} | {"Val Loss":^7} | {"Train_mAP":^7} | {"Val_mAP":^7} | {"test_mAP":^7} | {"Time, m":^6}
"""
header_ssl = f"""
 Epoch | {"Loss":^6}  | {"Time, m":^6}
"""
raw_line_ssl = "{:6d} | {:6.2f} | {:6.2f}"
raw_line = "{:6d} | {:7.3f} | {:7.3f} | {:7.3f} | {:6.2f}"
raw_line_split = "{:6d} | {:7.3f} | {:7.3f} | {:7.3f} | {:7.3f} | {:7.3f} | {:6.2f}"
temp_header = f"""
 Epoch | {"Loss":^6} | {"Val Loss":^7} | {"mAP":^7} | {"CmAP":^7} | {"future_mAP":^7} | {"future_CmAP":^7} | {"Time, m":^6}
"""
temp_raw_line = "{:6d} | {:7.3f} | {:7.3f} | {:7.3f} | {:7.3f} | {:7.3f} | {:7.3f} | {:6.2f}"

def cholect45_ivtmetrics_mAP(df, CFG):
    """
    Compute the official CholecT45 mAP score.
    read n_triple, map_file_path from CFG.
    """
    n_triple = getattr(CFG, "n_triple", 170)
    map_path = getattr(CFG, "map_file_path", "maps.txt")
    
    # Get the indexes of the 1st triplet/prediction columns
    tri0_idx = int(df.columns.get_loc("tri0"))
    pred0_idx = int(df.columns.get_loc("0"))

    # Initiate empty list to store the folds mAP
    ivt = []
    classwise_ap_list = [] 

    # Loop over the 5 folds
    for fold in range(CFG.n_fold):
        
        rec = Recognition(num_class=n_triple, map_file_path=map_path)

        # Filter the fold and its corresponding videos
        fold_df = df[df["fold"] == fold]
        vids = fold_df.video.unique()

        # Loop over the videos
        for i, v in enumerate(vids):
            # Filter the video
            vid_df = fold_df[fold_df["video"] == v]

            
            rec.update(
                vid_df.iloc[:, tri0_idx : tri0_idx + n_triple].values,
                vid_df.iloc[:, pred0_idx : pred0_idx + n_triple].values,
            )

            rec.video_end()
    
        # Get the final mAP score for the fold
        result = rec.compute_video_AP('ivt', ignore_null=CFG.ignore_null)
        ivt.append(result['mAP'])
        classwise_ap_list.append(result['AP'])

    # Return the mean mAP value over 5 folds
    return np.mean(ivt)

def cholect45_ivtmetrics_mAP_all(df, CFG):
    """
    Compute official CholecT45 mAP for all components.
    read n_triple, map_file_path from CFG.
    """
    n_triple = getattr(CFG, "n_triple", 170)
    map_path = getattr(CFG, "map_file_path", "maps.txt")
    
    tri0_idx = int(df.columns.get_loc("tri0"))
    pred0_idx = int(df.columns.get_loc("0"))

    components = ['i', 'v', 't', 'iv', 'it', 'ivt']
    mAPs = {comp: [] for comp in components}
    classwise_APs = {comp: [] for comp in components}

    for fold in tqdm(range(CFG.n_fold)):
        
        rec = Recognition(num_class=n_triple, map_file_path=map_path, ignore_null=CFG.ignore_null)
        fold_df = df[df["fold"] == fold]
        vids = fold_df.video.unique()

        for v in vids:
            vid_df = fold_df[fold_df["video"] == v]
            
            rec.update(
                vid_df.iloc[:, tri0_idx : tri0_idx + n_triple].values,
                vid_df.iloc[:, pred0_idx : pred0_idx + n_triple].values,
            )
            rec.video_end()

        for comp in components:
            result = rec.compute_video_AP(comp, ignore_null=CFG.ignore_null)
            mAPs[comp].append(result['mAP'])
            classwise_APs[comp].append(result['AP'])

    mean_mAPs = {comp: np.mean(mAPs[comp]) for comp in components}
    std_mAPs = {comp: np.std(mAPs[comp]) for comp in components}
    mean_classwise_APs = {comp: np.mean(classwise_APs[comp], axis=0) for comp in components}

    classwise_AP_dfs = {}
    for comp in components:
        classwise_AP_df = pd.DataFrame({
            "class": [f'class_{i}' for i in range(len(mean_classwise_APs[comp]))],
            "AP" : mean_classwise_APs[comp]
        })
        classwise_AP_dfs[comp] = classwise_AP_df

    return mean_mAPs, std_mAPs, classwise_AP_dfs
#===============================================================================================
def per_epoch_ivtmetrics_all(fold_df, CFG):
    """
    Compute per-epoch ivtmetrics. (called from train.py)
    read n_triple, map_file_path from CFG.
    """
    n_triple = getattr(CFG, "n_triple", 170)
    map_path = getattr(CFG, "map_file_path", "maps.txt")
    
    tri0_idx = int(fold_df.columns.get_loc("tri0"))
    pred0_idx = int(fold_df.columns.get_loc("0"))

    
    rec = Recognition(num_class=n_triple, map_file_path=map_path)
    vids = fold_df.video.unique()

    for i, v in enumerate(vids):
        vid_df = fold_df[fold_df["video"] == v]

        gt_array = vid_df.iloc[:, tri0_idx : tri0_idx + n_triple].values
        pred_array = vid_df.iloc[:, pred0_idx : pred0_idx + n_triple].values

        if np.issubdtype(gt_array.dtype, np.floating):
            gt_array = (gt_array >= 0.5).astype(int)

        rec.update(
            gt_array,
            pred_array,
        )
        rec.video_end()

    mAP = rec.compute_video_AP("ivt", ignore_null=CFG.ignore_null)["mAP"]
    i_mAP = rec.compute_video_AP("i", ignore_null=CFG.ignore_null)["mAP"]
    v_mAP = rec.compute_video_AP("v", ignore_null=CFG.ignore_null)["mAP"]
    t_mAP = rec.compute_video_AP("t", ignore_null=CFG.ignore_null)["mAP"]
    return mAP, i_mAP, v_mAP, t_mAP
#===============================================================================================


def per_epoch_ivtmetrics(fold_df, CFG):
    """
    Compute per-epoch ivtmetrics. (single 'ivt' mAP)
    read n_triple, map_file_path from CFG.
    """
    n_triple = getattr(CFG, "n_triple", 170)
    map_path = getattr(CFG, "map_file_path", "maps.txt")
    
    tri0_idx = int(fold_df.columns.get_loc("tri0"))
    pred0_idx = int(fold_df.columns.get_loc("0"))

    
    rec = Recognition(num_class=n_triple, map_file_path=map_path)
    vids = fold_df.video.unique()

    for i, v in enumerate(vids):
        vid_df = fold_df[fold_df["video"] == v]
        
        rec.update(
            vid_df.iloc[:, tri0_idx : tri0_idx + n_triple].values,
            vid_df.iloc[:, pred0_idx : pred0_idx + n_triple].values,
        )
        rec.video_end()

    mAP = rec.compute_video_AP("ivt", ignore_null=CFG.ignore_null)["mAP"]
    return mAP

def per_epoch_ivtmetrics_inst(fold_df, CFG):
    
    n_inst = getattr(CFG, "n_inst", 6)
    map_path = getattr(CFG, "map_file_path", "maps.txt")
    
    inst_cols = [f"inst{i}" for i in range(n_inst)]
    pred_cols = [str(i) for i in range(n_inst)]
    assert all(c in fold_df.columns for c in inst_cols), f"Missing inst0..inst{n_inst-1}"
    assert all(c in fold_df.columns for c in pred_cols), f"Missing pred '0'..'{n_inst-1}'"

    
    rec = Recognition(num_class=n_inst, map_file_path=map_path)

    for v in fold_df["video"].unique():
        vid_df = fold_df.loc[fold_df["video"] == v]
        targets  = vid_df[inst_cols].to_numpy(dtype=float)  # [T,6]
        predicts = vid_df[pred_cols].to_numpy(dtype=float)  # [T,6]
        rec.update(targets, predicts)
        rec.video_end()

    # No extract(): inputs are already 6-class
    mAP = rec.compute_video_AP_only(ignore_null=False)["mAP"]
    return float(mAP)

def per_epoch_ivtmetrics_verb(fold_df, CFG):
    
    n_verb = getattr(CFG, "n_verb", 10)
    map_path = getattr(CFG, "map_file_path", "maps.txt")
    
    v_cols   = [f"v{i}" for i in range(n_verb)]
    pred_cols = [str(i) for i in range(n_verb)]
    assert all(c in fold_df.columns for c in v_cols),   f"Missing v0..v{n_verb-1}"
    assert all(c in fold_df.columns for c in pred_cols), f"Missing pred '0'..'{n_verb-1}'"

    
    rec = Recognition(num_class=n_verb, map_file_path=map_path)

    for v in fold_df["video"].unique():
        vid = fold_df.loc[fold_df["video"] == v]
        y_true = vid[v_cols].to_numpy(dtype=float)    # [T,10]
        y_pred = vid[pred_cols].to_numpy(dtype=float) # [T,10]
        rec.update(y_true, y_pred)
        rec.video_end()

    return float(rec.compute_video_AP_only(ignore_null=False)["mAP"])


def per_epoch_ivtmetrics_target(fold_df, CFG):
    
    n_target = getattr(CFG, "n_target", 15)
    map_path = getattr(CFG, "map_file_path", "maps.txt")
    
    t_cols    = [f"t{i}" for i in range(n_target)]
    pred_cols = [str(i) for i in range(n_target)]
    assert all(c in fold_df.columns for c in t_cols),    f"Missing t0..t{n_target-1}"
    assert all(c in fold_df.columns for c in pred_cols), f"Missing pred '0'..'{n_target-1}'"

    
    rec = Recognition(num_class=n_target, map_file_path=map_path)

    for v in fold_df["video"].unique():
        vid = fold_df.loc[fold_df["video"] == v]
        y_true = vid[t_cols].to_numpy(dtype=float)    # [T,15]
        y_pred = vid[pred_cols].to_numpy(dtype=float) # [T,15]
        rec.update(y_true, y_pred)
        rec.video_end()

    return float(rec.compute_video_AP_only(ignore_null=False)["mAP"])

