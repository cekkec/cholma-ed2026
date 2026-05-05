#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preprocess_multi_baseline.py (FINAL)
- multi-annotator (2 or 3 annotators) baseline preprocess
- enforce meta columns right after CSV load (folder/frame/video/image_path/image_id)
- enforce label column set/order (fill with 0 if missing)
- merge per-frame labels in intersection/union mode
- null_triplet_index guard when intersection is empty
- add video_num / fold columns
- provide identity check util for 2~3 CSVs' column composition and order
"""

from __future__ import annotations
import os, re
from typing import Dict, List, Tuple, Iterable, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ------------------------------
# IO helpers
# ------------------------------
ENCODINGS = ["utf-8", "utf-8-sig", "cp949", "euc-kr"]

def read_csv_any(path: str) -> pd.DataFrame:
    last = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last = e
            continue
    # final attempt: no encoding
    return pd.read_csv(path)

# ------------------------------
# Meta normalization
# ------------------------------
def normalize_meta_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    - strip whitespace from column names
    - ensure image_path (build from folder/frame or video/frame if missing)
    - ensure folder, frame, video
    - ensure image_id (based on image_path)
    """
    df = df.copy()
    df.columns = df.columns.map(lambda c: str(c).strip())

    # image_path
    if "image_path" not in df.columns:
        if {"folder", "frame"}.issubset(df.columns):
            df["image_path"] = df["folder"].astype(str).str.strip() + "/" + df["frame"].astype(str).str.strip()
        elif {"video", "frame"}.issubset(df.columns):
            df["image_path"] = df["video"].astype(str).str.strip() + "/" + df["frame"].astype(str).str.strip()
        else:
            df["image_path"] = df.index.to_series().map(lambda i: f"row{i}")

    # folder
    if "folder" not in df.columns:
        df["folder"] = df["image_path"].astype(str).str.rsplit("/", n=1).str[0]

    # frame
    if "frame" not in df.columns:
        df["frame"] = df["image_path"].astype(str).str.rsplit("/", n=1).str[-1]

    # video
    if "video" not in df.columns:
        def _to_video(s):
            m = re.findall(r"\d+", str(s))
            return m[0] if m else str(s)
        df["video"] = df["folder"].map(_to_video)

    # image_id
    if "image_id" not in df.columns:
        df["image_id"] = df["image_path"].astype(str).str.strip()

    # type and whitespace cleanup
    for c in ["folder", "frame", "video", "image_path", "image_id"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df

def add_video_numeric(df: pd.DataFrame, video_col: str = "video", new_col: str = "video_num") -> pd.DataFrame:
    df = df.copy()
    def to_num(v):
        s = str(v)
        if s.isdigit():
            return int(s)
        m = re.findall(r"\d+", s)
        return int(m[0]) if m else -1
    df[new_col] = df[video_col].apply(to_num).astype(int)
    return df

# ------------------------------
# Label spec & ordering
# ------------------------------
@dataclass(frozen=True)
class LabelSpec:
    tri: int = 100
    inst: int = 6
    verb: int = 10
    target: int = 15
    IV: int = 60
    IT: int = 90
    VT: int = 150

    @property
    def tri_cols(self) -> List[str]:
        return [f"tri{i}" for i in range(self.tri)]

    @property
    def inst_cols(self) -> List[str]:
        return [f"inst{i}" for i in range(self.inst)]

    @property
    def verb_cols(self) -> List[str]:
        return [f"v{i}" for i in range(self.verb)]

    @property
    def target_cols(self) -> List[str]:
        return [f"t{i}" for i in range(self.target)]

    @property
    def IV_cols(self) -> List[str]:
        return [f"IV{i}" for i in range(self.IV)]

    @property
    def IT_cols(self) -> List[str]:
        return [f"IT{i}" for i in range(self.IT)]

    @property
    def VT_cols(self) -> List[str]:
        return [f"VT{i}" for i in range(self.VT)]

    @property
    def meta_cols(self) -> List[str]:
        return ["folder", "frame", "video", "image_path", "image_id"]

    @property
    def all_label_cols(self) -> List[str]:
        return self.tri_cols + self.inst_cols + self.verb_cols + self.target_cols + self.IV_cols + self.IT_cols + self.VT_cols

    @property
    def full_order(self) -> List[str]:
        
        return self.meta_cols + self.all_label_cols

LABEL_SPEC = LabelSpec()

def ensure_label_columns(df: pd.DataFrame, spec: LabelSpec = LABEL_SPEC) -> pd.DataFrame:
    """
    - fill label columns with 0 if missing
    - type: float32
    - order: reorder by spec.full_order, skipping non-existent
    """
    df = df.copy()
    
    for c in spec.all_label_cols:
        if c not in df.columns:
            df[c] = 0.0

    
    for cols in [spec.tri_cols, spec.inst_cols, spec.verb_cols, spec.target_cols, spec.IV_cols, spec.IT_cols, spec.VT_cols]:
        df[cols] = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0).astype(np.float32)

    ordered = [c for c in spec.full_order if c in df.columns]
    
    tail = [c for c in df.columns if c not in ordered]
    df = df[ordered + tail]
    return df

# ------------------------------
# Column equality checkers
# ------------------------------
def get_ordered_columns_for_check(df: pd.DataFrame, spec: LabelSpec = LABEL_SPEC) -> List[str]:
    """
    - column list for comparison: meta (5 items) + all labels (in spec order)
    - for identity check of column composition and order across datasets
    """
    cols = []
    for c in spec.meta_cols:
        if c in df.columns:
            cols.append(c)
        else:
            cols.append(f"[MISSING:{c}]")
    for c in spec.all_label_cols:
        if c in df.columns:
            cols.append(c)
        else:
            cols.append(f"[MISSING:{c}]")
    return cols

def print_column_diffs(name_a: str, df_a: pd.DataFrame, name_b: str, df_b: pd.DataFrame, spec: LabelSpec = LABEL_SPEC) -> None:
    cols_a = get_ordered_columns_for_check(df_a, spec)
    cols_b = get_ordered_columns_for_check(df_b, spec)

    same = (cols_a == cols_b)
    print(f"\n[COL-CHECK] {name_a} vs {name_b}")
    print(f"- same_order_and_presence: {same}")
    if not same:
        print("  * First 50 diffs (idx: A | B):")
        for i, (ca, cb) in enumerate(zip(cols_a, cols_b)):
            if ca != cb:
                print(f"    [{i:03d}] {ca} | {cb}")
            if i > 200:  # cut if too long
                print("    ... (truncated)")
                break

def check_annotator_columns(annot_dfs: Dict[str, pd.DataFrame], spec: LabelSpec = LABEL_SPEC) -> bool:
    """
    Check whether the given DFs (2~3 items) have identical column composition and order;
    print diff if different. Returns True if all identical, else False.
    """
    names = list(annot_dfs.keys())
    if len(names) < 2:
        print("[COL-CHECK] only one dataframe given; nothing to compare.")
        return True
    ok_all = True
    base_name = names[0]
    base_df = annot_dfs[base_name]
    base_cols = get_ordered_columns_for_check(base_df, spec)
    for nm in names[1:]:
        cur_cols = get_ordered_columns_for_check(annot_dfs[nm], spec)
        same = (base_cols == cur_cols)
        if not same:
            ok_all = False
            print_column_diffs(base_name, base_df, nm, annot_dfs[nm], spec)
    if ok_all:
        print(f"[COL-CHECK] All annotators have identical column composition and order: {names}")
    return ok_all

# ------------------------------
# Fusing (union / intersection)
# ------------------------------
def _binary_union(arrs: List[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(arrs[0], dtype=np.float32)
    for a in arrs:
        out = np.maximum(out, (a > 0).astype(np.float32))
    return out

def _binary_intersection(arrs: List[np.ndarray]) -> np.ndarray:
    out = np.ones_like(arrs[0], dtype=np.float32)
    for a in arrs:
        out = np.minimum(out, (a > 0).astype(np.float32))
    return out

def _pick_fuse_fn(mode: str):
    mode = str(mode).strip().lower()
    if mode == "union":
        return _binary_union
    if mode == "intersection":
        return _binary_intersection
    raise ValueError(f"Unknown fuse mode: {mode}. Use 'union' or 'intersection'.")

def fuse_rows(rows: List[pd.Series], mode: str, null_triplet_index: int, spec: LabelSpec = LABEL_SPEC) -> Dict[str, np.ndarray]:
    """
    Collect labels from different annotators for the same image_id (= same frame),
    then merge with union/intersection.
    - if intersection result has all tri = 0, set null_triplet_index position to 1 (when >= 0)
    """
    fuse = _pick_fuse_fn(mode)

    def stack(cols: List[str]) -> np.ndarray:
        mats = []
        for r in rows:
            v = r[cols].to_numpy(dtype=np.float32, copy=False)
            mats.append(v[np.newaxis, :])
        return np.concatenate(mats, axis=0)  # (N_annot, C)

    # merge
    tri = fuse([row[spec.tri_cols].to_numpy(dtype=np.float32) for row in rows])
    inst = fuse([row[spec.inst_cols].to_numpy(dtype=np.float32) for row in rows])
    verb = fuse([row[spec.verb_cols].to_numpy(dtype=np.float32) for row in rows])
    targ = fuse([row[spec.target_cols].to_numpy(dtype=np.float32) for row in rows])
    IV   = fuse([row[spec.IV_cols].to_numpy(dtype=np.float32)   for row in rows])
    IT   = fuse([row[spec.IT_cols].to_numpy(dtype=np.float32)   for row in rows])
    VT   = fuse([row[spec.VT_cols].to_numpy(dtype=np.float32)   for row in rows])

    # when intersection empty null-triplet guard
    if mode == "intersection" and null_triplet_index is not None and null_triplet_index >= 0:
        if tri.sum() == 0:
            if 0 <= null_triplet_index < len(tri):
                tri[null_triplet_index] = 1.0

    return {
        "tri": tri, "inst": inst, "verb": verb, "target": targ,
        "IV": IV, "IT": IT, "VT": VT
    }

# ------------------------------
# Video2Fold helpers
# ------------------------------
DEFAULT_FOLD_MAP = {
    "cholect50-crossval": {
        0: [79,  2, 51,  6, 25, 14, 66, 23, 50, 111],
        1: [80, 32,  5, 15, 40, 47, 26, 48, 70,  96],
        2: [31, 57, 36, 18, 52, 68, 10,  8, 73, 103],
        3: [42, 29, 60, 27, 65, 75, 22, 49, 12, 110],
        4: [78, 43, 62, 35, 74,  1, 56,  4, 13,  92],
    }
}

def build_video2fold_from_schema(schema_name: str = "cholect50-crossval") -> Dict[str, int]:
    if schema_name not in DEFAULT_FOLD_MAP:
        raise KeyError(f"Unknown fold schema: {schema_name}")
    mp: Dict[str, int] = {}
    table = DEFAULT_FOLD_MAP[schema_name]
    for f, vids in table.items():
        for v in vids:
            mp[str(v)] = int(f)
    return mp

# ------------------------------
# Core builder
# ------------------------------
def load_and_harmonize_one(path: str, annot_name: str, spec: LabelSpec = LABEL_SPEC) -> pd.DataFrame:
    df_raw = read_csv_any(path)
    df_norm = normalize_meta_columns(df_raw)
    df_out = ensure_label_columns(df_norm, spec)
    # print(f"[{annot_name}] head:", df_out.head(1)[["image_path","image_id"]])
    return df_out

def build_fused_full_dataframe(CFG, video2fold: Optional[Dict[str, int]] = None,
                               spec: LabelSpec = LABEL_SPEC) -> pd.DataFrame:
    """
    CFG.multi_label:
      - enabled: true
      - annotators: ["Anno1","Anno2","GT"]  # only 2 items for a 2-annotator experiment
      - csv_paths: {GT: ..., Anno1: ..., Anno2: ...}
      - train_val_mode: "intersection" | "union"
      - null_triplet_index: 0 (or -1)
    """
    assert hasattr(CFG, "multi_label") and CFG.multi_label.enabled, "[multi-baseline] CFG.multi_label.enabled must be true"
    annos: List[str] = list(CFG.multi_label.annotators)
    csv_mp: Dict[str, str] = dict(CFG.multi_label.csv_paths)
    mode: str = str(CFG.multi_label.train_val_mode).strip().lower()
    null_idx: int = int(getattr(CFG.multi_label, "null_triplet_index", -1))

    dfs: Dict[str, pd.DataFrame] = {}
    for nm in annos:
        if nm not in csv_mp:
            raise KeyError(f"[multi-baseline] annotator '{nm}' has no path in multi_label.csv_paths")
        p = csv_mp[nm]
        if not os.path.exists(p):
            raise FileNotFoundError(f"[multi-baseline] CSV not found for {nm}: {p}")
        dfs[nm] = load_and_harmonize_one(p, nm, spec)

    
    _ok = check_annotator_columns(dfs, spec)

    
    
    # map each annotator df to image_id -> row series
    maps: Dict[str, Dict[str, pd.Series]] = {}
    for nm, df in dfs.items():
        maps[nm] = {row["image_id"]: row for _, row in df.iterrows()}

    all_keys = sorted({k for m in maps.values() for k in m.keys()})

    # fuse
    fused_rows = []
    for key in all_keys:
        # collect only annotator rows for this key
        rows = []
        avail_names = []
        for nm in annos:
            r = maps[nm].get(key, None)
            if r is not None:
                rows.append(r)
                avail_names.append(nm)

        if not rows:
            continue  # theoretically none

        # metasafely extract from first row
        r0 = rows[0]
        meta = {}
        for k in spec.meta_cols:
            if k in r0.index:
                meta[k] = r0[k]
            else:
                if k == "image_id":
                    meta[k] = r0.get("image_path", f"row{getattr(r0,'name', -1)}")
                elif k == "image_path":
                    f = r0.get("folder", "")
                    fr = r0.get("frame", "")
                    meta[k] = f"{f}/{fr}" if (f or fr) else f"row{getattr(r0,'name', -1)}"
                elif k == "video":
                    meta[k] = r0.get("video", r0.get("folder", ""))
                else:
                    meta[k] = r0.get(k, "")

        fused = fuse_rows(rows, mode=mode, null_triplet_index=null_idx, spec=spec)

        out = {
            "folder": meta["folder"],
            "frame":  meta["frame"],
            "video":  meta["video"],
            "image_path": meta["image_path"],
            "image_id": meta["image_id"],
        }
        # stack back
        for i, col in enumerate(spec.tri_cols):
            out[col] = float(fused["tri"][i])
        for i, col in enumerate(spec.inst_cols):
            out[col] = float(fused["inst"][i])
        for i, col in enumerate(spec.verb_cols):
            out[col] = float(fused["verb"][i])
        for i, col in enumerate(spec.target_cols):
            out[col] = float(fused["target"][i])
        for i, col in enumerate(spec.IV_cols):
            out[col] = float(fused["IV"][i])
        for i, col in enumerate(spec.IT_cols):
            out[col] = float(fused["IT"][i])
        for i, col in enumerate(spec.VT_cols):
            out[col] = float(fused["VT"][i])

        fused_rows.append(out)

    fused = pd.DataFrame(fused_rows)

    if fused.empty:
        raise RuntimeError("[multi-baseline] fused DataFrame is empty; check annotator CSV alignment and keys.")

    
    fused = normalize_meta_columns(fused)
    fused = ensure_label_columns(fused, spec)
    fused = add_video_numeric(fused, video_col="video", new_col="video_num")

    # fold assign
    if video2fold is not None:
        def _to_fold(v: str) -> int:
            v = str(v)
            if v in video2fold:
                return int(video2fold[v])
            # extract number for matching just in case
            m = re.findall(r"\d+", v)
            if m and (m[0] in video2fold):
                return int(video2fold[m[0]])
            return -1
        fused["fold"] = fused["video"].map(_to_fold).astype(int)
    else:
        fused["fold"] = -1

    print(f"[multi-baseline] annotators: {annos}")
    print(f"[multi-baseline] mode: {mode}, frames: {len(fused)}, by fold: {dict(fused['fold'].value_counts().sort_index())}")
    print(f"[multi-baseline] null_triplet_index={null_idx} ({'used for empty-intersection guard' if (mode=='intersection' and null_idx>=0) else 'unused'})")

    return fused

# ------------------------------
# CLI (optional quick test)
# ------------------------------
if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotators", type=str, default='["Anno1","Anno2","GT"]',
                    help='JSON list, e.g. \'["Anno1","Anno2"]\'')
    ap.add_argument("--csv_paths", type=str, required=True,
                    help='JSON dict, e.g. \'{"GT":".../GT.csv","Anno1":".../A1.csv","Anno2":".../A2.csv"}\'')
    ap.add_argument("--mode", type=str, default="intersection", choices=["intersection","union"])
    ap.add_argument("--null_triplet_index", type=int, default=0)
    ap.add_argument("--fold_schema", type=str, default="cholect50-crossval")
    ap.add_argument("--out_csv", type=str, default="./fused_multi.csv")
    args = ap.parse_args()

    annotators = json.loads(args.annotators)
    csv_paths  = json.loads(args.csv_paths)

    class _CFG: pass
    CFG = _CFG()
    CFG.multi_label = _CFG()
    CFG.multi_label.enabled = True
    CFG.multi_label.annotators = annotators
    CFG.multi_label.csv_paths = csv_paths
    CFG.multi_label.train_val_mode = args.mode
    CFG.multi_label.null_triplet_index = args.null_triplet_index

    v2f = build_video2fold_from_schema(args.fold_schema)
    fused = build_fused_full_dataframe(CFG, video2fold=v2f)
    fused.to_csv(args.out_csv, index=False)
    print(f"[DONE] saved fused CSV -> {os.path.abspath(args.out_csv)}")
