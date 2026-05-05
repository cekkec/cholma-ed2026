#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An python implementation triplet component filtering .
Created on Thu Dec 30 12:37:56 2021
@author: nwoye chinedu i.
(c) camma, icube, unistra
"""
#%%%%%%%% imports %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import sys

#%%%%%%%%% COMPONENT FILTER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Disentangle(object):
    """
    Class: filter a triplet prediction into the components (such as instrument i, verb v, target t, instrument-verb iv, instrument-target it, etc)     
    @args
    ----
        map_file_path: str. path to the dictionary map file (e.g., output/maps.txt)
    @params
    ----------
    bank :   2D array
        holds the dictionary mapping of all components    
    @methods
    ----------
    extract(input, componet): 
        call filter a component labels from the inputs labels     
    """

    def __init__(self, map_file_path="maps.txt"):
        try:
            
            self.bank = np.genfromtxt(map_file_path, dtype=int, comments='#', delimiter=',', skip_header=0)
            if self.bank.size == 0:
                print(f"!!! [Disentangle ERROR] bank is empty after loading '{map_file_path}'. (file empty or header only)")
                raise ValueError("np.genfromtxt returned an empty array.")
        except Exception as e:
            print(f"!!! [Disentangle ERROR] could not read file '{map_file_path}'.")
            print(f"    {e}")
            print("     [Disentangle WARN] map bank is empty; returning empty value.")
            self.bank = np.array([])  # init as empty array
        
    def decompose(self, inputs, component):
        """ Extract the component labels from the triplets.
            @args:
                inputs: a 1D vector of dimension (n), where n = number of triplet classes;
                        with values int(0 or 1) for target labels and float[0, 1] for predicted labels.
                component: a string for the component to extract; 
                        (e.g.: i, v, t, iv, it, vt)
            @return:
                output: int or float sparse encoding 1D vector of dimension (n_comp), where n_comp = number of component's classes.
        """
        
        # 0:IVT, 1:I, 2:V, 3:T, 4:IV, 5:IT, 6:VT
        txt2id = {'ivt':0, 'i':1, 'v':2, 't':3, 'iv':4, 'it':5, 'vt':6} 
        
        if component not in txt2id:
            sys.exit(f"Component '{component}' is not supported in [disentangle].")
        if self.bank.size == 0:
            
            return []  # return empty list on map load failure
        
        key    = txt2id[component]
        
        index  = sorted(np.unique(self.bank[:,key]))
        
        index = [i for i in index if i >= 0] 
        
        output = []
        for idx in index:
            
            
            
            
            rows_with_idx = (self.bank[:, key] == idx)
            
            
            same_class_triplet_indices = self.bank[rows_with_idx, 0]
            
            
            valid_indices = [i for i in same_class_triplet_indices if 0 <= i < len(inputs)]
            
            if not valid_indices:
                output.append(0.0)  # no triplet maps to this component
            else:
                y = np.max(np.array(inputs[valid_indices]))
                output.append(y) 
        
        return output
    
    def extract(self, inputs, component="i"):
        """
        Extract a component label from the triplet label
        @args
        ----
        inputs: 2D array, (Batch, n_triple) e.g., (Batch, 170)
            triplet labels, either predicted label or the groundtruth
        component: str,
            the symbol of the component to extract, (i, v, t, iv, it, vt)
        @return
        ------
        label: 2D array, (Batch, n_component_class) e.g., (Batch, 6)
            filtered component's labels
        """       
        if component == "ivt":
            return inputs
        else:
            component_list = [component]* len(inputs)
            
            #           ㄴ inputs: (Batch, 170)
            #           ㄴ decompose -> output: (6,) or (10,) ...
            
            return np.array(list(map(self.decompose, inputs, component_list)))

    