#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import timm

class SwinIVTModel(nn.Module):
    def __init__(self, CFG, model_name: str, pretrained: bool = True):
        super().__init__()
        
        
        self.n_inst   = getattr(CFG, "n_inst", 6)
        self.n_verb   = getattr(CFG, "n_verb", 10)
        self.n_target = getattr(CFG, "n_target", 15)
        self.n_triple = getattr(CFG, "n_triple", 170) # use 170 as-is

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        feat_dim = self.backbone.num_features

        self.i_head   = nn.Linear(feat_dim, self.n_inst)
        self.v_head   = nn.Linear(feat_dim, self.n_verb)
        self.t_head   = nn.Linear(feat_dim, self.n_target)
        
        self.ivt_head = nn.Linear(feat_dim, self.n_triple) 

    def forward(self, x):
        fvec = self.backbone(x)
        return {
            "i":   self.i_head(fvec),
            "v":   self.v_head(fvec),
            "t":   self.t_head(fvec),
            "ivt": self.ivt_head(fvec), # (BatchSize, 170) output
        }