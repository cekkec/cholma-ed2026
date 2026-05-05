#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# config_summary.py
import os
import sys
from datetime import datetime

def _bool(v):  # safe conversion
    try:
        return bool(v)
    except Exception:
        return v

def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)

def _extract_cli_overrides():
    
    
    args = []
    for a in sys.argv[1:]:
        if "=" in a and not a.startswith("-"):
            args.append(a)
    return args

def _folds_applied(CFG):
    trn = list(getattr(CFG, "trn_fold", []))
    n_fold = int(getattr(CFG, "n_fold", 1))
    active = [f for f in trn if isinstance(f, int) and f < n_fold]
    ignored = [f for f in trn if f not in active]
    return active, ignored

def _make_box(title, lines):
    min_width = 30 
    width = max(min_width, len(title) + 4, *(len(l) for l in lines)) + 4
    top = "┏" + "━" * (width - 2) + "┓"
    mid_title = f" {title} "
    pad_title = "━" * max(0, width - 2 - len(mid_title))
    title_line = f"┃{mid_title}{pad_title}┫"
    body = [f"┃ {_}"+(" " * (width - 3 - len(_)))+"┃" for _ in lines]
    bottom = "┗" + "━" * (width - 2) + "┛"
    return "\n".join([top, title_line, *body, bottom])

def print_config_summary(CFG):
    """Pretty-print to console the settings actually applied to the current run."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cli = _extract_cli_overrides()

    model_type = str(getattr(CFG, "model_type", "default")).lower()
    use_graph = (model_type == "graph")
    warmup_epochs      = int(getattr(CFG, "warmup_epochs", 0))
    stage3_start_epoch = int(getattr(CFG, "stage3_start_epoch", 10**9))
    force_stage        = getattr(CFG, "force_stage", None)
    cam_vis_enable = _bool(getattr(CFG, "cam_vis_enable", False))
    ema_on = _bool(getattr(CFG, "ema_teacher", use_graph))
    ema_decay = getattr(CFG, "ema_decay", 0.999)
    active_folds, ignored_folds = _folds_applied(CFG)

    # --- [★ key change 1] ---
    
    lines_main = [
        f"Datetime       : {now}",
        f"Device         : {getattr(CFG, 'device', 'cuda:0')}",
        f"Seed           : {getattr(CFG, 'seed', 42)}",
        f"Exp            : {getattr(CFG, 'exp', '')}",
        f"Output Dir     : {getattr(CFG, 'output_dir', '')}",
        
        f"Train CSV      : {os.path.basename(getattr(CFG, 'train_csv_path', 'N/A'))}",
        f"Val CSV        : {os.path.basename(getattr(CFG, 'val_csv_path', 'N/A'))}",
    ]
    print(_make_box("RUN CONTEXT", lines_main))

    lines_model = [
        f"Model Type       : {model_type}",
        f"Backbone         : {getattr(CFG, 'model_name', '')}",
        f"Pretrained       : {_bool(getattr(CFG, 'pretrained', True))}",
        
        f"Classes (I/V/T)  : {getattr(CFG, 'n_inst', '-')}/{getattr(CFG, 'n_verb', '-')}/{getattr(CFG, 'n_target', '-')}",
        f"Classes (Triple) : {getattr(CFG, 'n_triple', '-')} (Null Idx: {getattr(CFG, 'null_triplet_index', '-')})",
        f"Img Size (HxW)   : {getattr(CFG, 'height', 224)} x {getattr(CFG, 'width', 224)}",
    ]
    print(_make_box("MODEL", lines_model))

    # --- [★ key change 2] ---
    lines_schedule = [
        f"Epochs           : {getattr(CFG, 'epochs', 1)}",
        
        f"Val Mode (Filter): {getattr(CFG, 'val_data_mode', 'normal')}",
        f"Batch Size (Tr/Va): {getattr(CFG, 'batch_size', 32)} / {getattr(CFG, 'valid_batch_size', 32)}",
        f"Grad Accum Steps : {getattr(CFG, 'gradient_accumulation_steps', 1)}",
        f"Workers          : {getattr(CFG, 'nworkers', 4)}",
        f"Shuffle          : {_bool(getattr(CFG, 'shuffle', True))}",
    ]
    print(_make_box("SCHEDULE", lines_schedule))

    lines_opt = [
        f"Optimizer        : {getattr(CFG, 'optim', 'adamw')}",
        f"LR / MinLR       : {_fmt(getattr(CFG,'lr',2e-4))} / {_fmt(getattr(CFG,'min_lr',2e-5))}",
        f"Weight Decay     : {_fmt(getattr(CFG,'weight_decay',1e-6))}",
        f"Cosine T_0       : {getattr(CFG, 'T_0', 21)}",
        f"Max Grad Norm    : {_fmt(getattr(CFG, 'max_grad_norm', 0.0))}",
    ]
    print(_make_box("OPTIMIZATION", lines_opt))

    lines_graph = [
        f"EMA Teacher      : {ema_on} (decay={ema_decay})",
        f"CAM→Nodes thr/γ  : {_fmt(getattr(CFG,'cam_threshold',0.25))} / {_fmt(getattr(CFG,'cam_gamma',1.5))}",
        f"Pseudo Top-K     : {getattr(CFG,'pseudo_topk_per_class',2)}",
        f"Nodes/Edges cap  : {getattr(CFG,'nodes_max_per_image',16)} / {getattr(CFG,'edges_max_per_image',64)}",
        f"MIL τ            : {_fmt(getattr(CFG,'mil_tau',0.5))}",
        f"λ(img), λ(verb)  : {_fmt(getattr(CFG,'lambda_img',0.5))}, {_fmt(getattr(CFG,'lambda_verb',1.0))}",
    ]
    if use_graph:
        print(_make_box("GRAPH / CAM / MIL", lines_graph))

    lines_viz = [
        f"Enable           : {cam_vis_enable}",
        f"Every (epochs)   : {getattr(CFG,'cam_vis_every', 1)}",
        f"Samples per epoch: {getattr(CFG,'cam_vis_n', 15)}",
        f"Pred thresh      : {_fmt(getattr(CFG,'cam_vis_pred_thr', 0.5))}",
    ]
    if cam_vis_enable:
        print(_make_box("CAM VIS", lines_viz))

    lines_split = [
        f"Split Selector   : {getattr(CFG, 'split_selector', '')}",
        f"n_fold / start   : {getattr(CFG,'n_fold',1)} / {getattr(CFG,'start_n_fold',0)}",
        f"trn_fold (raw)   : {list(getattr(CFG,'trn_fold', []))}",
        f"Active folds     : {active_folds}",
        f"Ignored folds    : {ignored_folds}",
    ]
    print(_make_box("DATA SPLIT", lines_split))

    cli_str = ", ".join(_extract_cli_overrides()) or "(none)"
    print(_make_box("CLI OVERRIDES", [cli_str]))