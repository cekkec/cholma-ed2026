import os
import pandas as pd
from sklearn.model_selection import GroupKFold
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from hydra.utils import to_absolute_path

def split_selector( case='cholect50'):
    switcher = {
        'cholect45-crossval': {
            1: [79,  2, 51,  6, 25, 14, 66, 23, 50,],
            2: [80, 32,  5, 15, 40, 47, 26, 48, 70,],
            3: [31, 57, 36, 18, 52, 68, 10,  8, 73,],
            4: [42, 29, 60, 27, 65, 75, 22, 49, 12,],
            5: [78, 43, 62, 35, 74,  1, 56,  4, 13,],
        },
         'cholect50-crossval': {
                1: [79,  2, 51,  6, 25, 14, 66, 23, 50, 111],
                2: [80, 32,  5, 15, 40, 47, 26, 48, 70,  96],
                3: [31, 57, 36, 18, 52, 68, 10,  8, 73, 103],
                4: [42, 29, 60, 27, 65, 75, 22, 49, 12, 110],
                5: [78, 43, 62, 35, 74,  1, 56,  4, 13,  92],
            },
    }
    return switcher.get(case)

def get_folds(CFG):
    print("\033[94mPreparing the data\033[0m")
    # Read the dataframe
    if getattr(CFG, "multi_label", None) and CFG.multi_label.get("enabled", False):
        csv_path = CFG.multi_label["csv_paths"]["GT"]
    else:
        # csv_path = CFG.path_csv
        csv_path = CFG.train_csv_path

    train = pd.read_csv(to_absolute_path(csv_path))


    print("Preprocessing the data...")

    # Start a folds df to map the folds
    folds = train.copy()
    fold_map = split_selector(CFG.split_selector)

    # Initialize the fold column
    folds["fold"] = -1
    
    # Assign each video to a fold based on the predefined lists in fold_map
    for fold, video_list in fold_map.items():
        video_list = [f"VID{vid:02d}" for vid in video_list]
        if isinstance(fold, int):
            folds.loc[folds["video"].isin(video_list), "fold"] = fold - 1
        else:
            folds.loc[folds["video"].isin(video_list), "fold"] = fold

    print("Dataset ready!\n")
    wrong_idx = folds[folds["fold"] != -1].index
    folds = folds.loc[wrong_idx].reset_index(drop=True)

    return folds