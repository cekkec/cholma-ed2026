#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import Sampler
import random 
from collections import defaultdict, Counter
from functools import lru_cache
import ast
import pickle
import re 

cv2.setNumThreads(1)

class TrainDataset(Dataset):
    def __init__(self, df, CFG, transform=None, fold=None, inference=False, mode='normal'):
        """
        - null_triplet_index logic removed entirely.
        - n_triple (170) read from config; loads all columns 0..169.
        - image_base_dir replaced by parent_path/train_path.
        """
        self.CFG = CFG
        self.transform = transform
        self.inference = inference
        
        
        if mode == 'intersection':
            
            n_triple = getattr(CFG, "n_triple", 170)
            
            # select only valid label columns
            label_cols = [c for c in df.columns if re.match(r'^(inst|v|t)\d+$', c)]
            
            label_cols += [f"tri{i}" for i in range(n_triple) if f"tri{i}" in df.columns]

            if not label_cols:
                print("   [WARN] 'intersection' mode but no valid label columns found.")
                self.df = df.copy()
            else:
                # keep only rows where sum of valid labels >= 1
                has_label = (df[label_cols].sum(axis=1) > 0)
                self.df = df[has_label].reset_index(drop=True)
        else:
            self.df = df.copy()
        
        self.file_names = self.df["image_path"].values

        
        
        n_triple = getattr(CFG, "n_triple", 170)
        n_inst   = getattr(CFG, "n_inst", 6)
        n_verb   = getattr(CFG, "n_verb", 10)
        n_target = getattr(CFG, "n_target", 15)

        
        self.tri_cols    = [f"tri{i}" for i in range(n_triple)]
        self.inst_cols   = [f"inst{i}" for i in range(n_inst)]
        self.verb_cols   = [f"v{i}" for i in range(n_verb)]
        self.target_cols = [f"t{i}" for i in range(n_target)]
        
        # (IV, IT, VT) - metadata
        self.iv_cols = sorted([c for c in self.df.columns if re.match(r'^IV\d+$', c)])
        self.it_cols = sorted([c for c in self.df.columns if re.match(r'^IT\d+$', c)])
        self.vt_cols = sorted([c for c in self.df.columns if re.match(r'^VT\d+$', c)])
        
        # Ncount check (e.g., 170, 6)
        self.n_triple = len(self.tri_cols)
        self.n_inst   = len(self.inst_cols)
        self.n_verb   = len(self.verb_cols)
        self.n_target = len(self.target_cols)
        
        # data load (now loads all 170 / 6)
        try:
            self.triplet_label = torch.FloatTensor(self.df[self.tri_cols].values.astype(np.float16))
            self.inst_label    = torch.FloatTensor(self.df[self.inst_cols].values.astype(np.float16))
            self.verb_label    = torch.FloatTensor(self.df[self.verb_cols].values.astype(np.float16))
            self.target_label  = torch.FloatTensor(self.df[self.target_cols].values.astype(np.float16))
        except KeyError as e:
            print(f"[ERROR] Required column missing in DataFrame: {e}")
            print(f"   (e.g.: {self.tri_cols[-1]} or {self.inst_cols[-1]} missing)")
            print(f"    CFG.n_triple({n_triple}), CFG.n_inst({n_inst}) ensure settings match CSV file.")
            raise e
        except ValueError as e:
            print(f"[ERROR] Label column contains non-numeric values (e.g., 'triplet_ids'): {e}")
            raise e

        self.iv_label = torch.FloatTensor(self.df[self.iv_cols].values.astype(np.float16)) if self.iv_cols else torch.zeros((len(self.df), 1))
        self.it_label = torch.FloatTensor(self.df[self.it_cols].values.astype(np.float16)) if self.it_cols else torch.zeros((len(self.df), 1))
        self.vt_label = torch.FloatTensor(self.df[self.vt_cols].values.astype(np.float16)) if self.vt_cols else torch.zeros((len(self.df), 1))

        
        self.tri_count_cols    = [f"tri{i}_count" for i in range(n_triple)]
        self.inst_count_cols   = [f"inst{i}_count" for i in range(n_inst)]
        self.verb_count_cols   = [f"v{i}_count" for i in range(n_verb)]
        self.target_count_cols = [f"t{i}_count" for i in range(n_target)]

        
        self.triplet_count = self._load_or_create_zeros(self.tri_count_cols, self.n_triple)
        self.inst_count    = self._load_or_create_zeros(self.inst_count_cols, self.n_inst)
        self.verb_count    = self._load_or_create_zeros(self.verb_count_cols, self.n_verb)
        self.target_count  = self._load_or_create_zeros(self.target_count_cols, self.n_target)

    def _load_or_create_zeros(self, count_cols, n_labels):
        
        valid_count_cols = [c for c in count_cols if c in self.df.columns]
        
        if valid_count_cols:
            
            if len(valid_count_cols) != n_labels:
                print(f"[WARN] _load_or_create_zeros: expected Count column count({n_labels}) vs")
                print(f"       columns existing in CSV({len(valid_count_cols)})differs.")
            
            return torch.FloatTensor(self.df[valid_count_cols].values.astype(np.float16))
        else:
            # CSV with no count columns at all (e.g. GT_processed)case
            return torch.zeros((len(self.df), n_labels), dtype=torch.float16)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        
        # 0/1 Labels (170, 6, 10, 15)
        ivt = self.triplet_label[index]
        inst = self.inst_label[index]
        verb = self.verb_label[index]
        target = self.target_label[index]
        it = self.it_label[index]
        iv = self.iv_label[index]
        vt = self.vt_label[index]
        
        # Count Labels
        ivt_count = self.triplet_count[index]
        inst_count = self.inst_count[index]
        verb_count = self.verb_count[index]
        target_count = self.target_count[index]

        
        if not hasattr(self.CFG, "parent_path") or not hasattr(self.CFG, "train_path"):
             raise AttributeError("[CONFIG ERROR] 'parent_path'/'train_path' missing in config.yaml.")
             
        file_path = os.path.join(self.CFG.parent_path, self.CFG.train_path, file_name)
        image = cv2.imread(file_path)

        if image is None:
            print("\n" + "="*50)
            print(f"[FATAL ERROR] cv2.imread failed to read file (returned None).")
            print(f"  - path: {file_path}")
            print(f"  - cause: 1. CFG.parent_path/train_path is wrong")
            print(f"          (Base: '{self.CFG.parent_path}', Sub: '{self.CFG.train_path}')")
            print(f"          2. CSV 'image_path' is wrong ('{file_name}')")
            print("="*50 + "\n")
            raise FileNotFoundError(f"imread failed for {file_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.inference:
            return image
        else:
            labels_tuple = (ivt, inst, verb, target, it, iv, vt)
            counts_tuple = (ivt_count, inst_count, verb_count, target_count)
            return image, labels_tuple, counts_tuple