#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An python implementation recognition AP for surgical action triplet evaluation.
Created on Thu Dec 30 12:37:56 2021
@author: nwoye chinedu i.
(c) icube, unistra
"""
#%%%%%%%% imports %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
from sklearn.metrics import average_precision_score
import warnings
import sys
from ivtmetrics.disentangle import Disentangle

#%%%%%%%%%% RECOGNITION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Recognition(Disentangle):
    """
    Class: compute (mean) Average Precision     
    @args
    ----
        num_class: int, optional. The number of class of the classification task (default = 100)
        map_file_path: str. path to the dictionary map file (e.g., output/maps.txt)
    @attributes
    ----------
    ...
    """     
    def __init__(self, num_class=100, map_file_path="maps.txt", ignore_null=False):
        
        super(Recognition, self).__init__(map_file_path=map_file_path)
        np.seterr(divide='ignore', invalid='ignore')
        self.num_class = num_class
        self.ignore_null = ignore_null
        self.reset_global()   

    def resolve_nan(self, classwise):
        classwise[classwise==-0.0] = np.nan
        return classwise
        
    ##%%%%%%%%%%%%%%%%%%% RESET OP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def reset(self):
        "call at the beginning of new experiment or epoch to reset the accumulators for preditions and groundtruths."
        self.predictions = np.empty(shape = [0,self.num_class], dtype=np.float32)
        self.targets     = np.empty(shape = [0,self.num_class], dtype=np.int32)        
        
    def reset_global(self):
        "call at the beginning of new experiment"
        self.global_predictions = []
        self.global_targets     = []
        self.reset()    
    
    ##%%%%%%%%%%%%%%%%%%% UPDATE OP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
    def update(self, targets, predictions):
        """
        update prediction function
        """
        self.predictions = np.append(self.predictions, predictions, axis=0)
        self.targets     = np.append(self.targets, targets, axis=0)        
        
    def video_end(self):
        "call to signal the end of current video. Needed during inference to log performance per video"        
        self.global_predictions.append(self.predictions)
        self.global_targets.append(self.targets)
        self.reset()
    
    ##%%%%%%%%%%%%%%%%%%% COMPUTE OP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def compute_AP(self, component="ivt", ignore_null=False):
        """
        compute performance for all seen examples after a reset()
        """
        if component in ["ivt", "it", "iv", "t", "v", "i"]:
            
            targets  = self.extract(self.targets, component)
            predicts = self.extract(self.predictions, component)
        else:
            sys.exit("Function filtering {} not yet supported!".format(component))
        
        if targets.size == 0 or predicts.size == 0:
             
             print(f"[WARN] compute_AP({component}) received empty array. Returning mAP 0.0.")
             return {"AP": np.array([]), "mAP": 0.0}

        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='[info] triplet classes not represented in this test sample will be reported as nan values.')            
            classwise = average_precision_score(targets, predicts, average=None)
            classwise = self.resolve_nan(classwise)
            
            if (ignore_null and component=="ivt"): 
                try:
                    n_inst = len(np.unique(self.bank[:, 1])) - 1  # exclude -1 (null)
                except:
                    n_inst = 6  # fallback to 6 on failure
                classwise = classwise[:-n_inst]
            mean      = np.nanmean(classwise)
        return {"AP":classwise, "mAP":mean}
    
    def compute_global_AP(self, component="ivt", ignore_null=False):
        """
        compute performance for all seen examples after a reset_global()
        """         
        global_targets      = self.global_targets
        global_predictions  = self.global_predictions
        if len(self.targets) > 0:
            global_targets.append(self.targets)
            global_predictions.append(self.predictions)
        targets  = np.concatenate(global_targets, axis=0)
        predicts = np.concatenate(global_predictions, axis=0)
        if component in ["ivt", "it", "iv", "t", "v", "i"]:
            targets  = self.extract(targets, component)
            predicts = self.extract(predicts, component)
        else:
            sys.exit("Function filtering {} not yet supported!".format(component))
            
        if targets.size == 0 or predicts.size == 0:
             print(f"[WARN] compute_global_AP({component}) received empty array. Returning mAP 0.0.")
             return {"AP": np.array([]), "mAP": 0.0}
             
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='[info] triplet classes not represented in this test sample will be reported as nan values.')            
            classwise = average_precision_score(targets, predicts, average=None)
            classwise = self.resolve_nan(classwise)
            if (ignore_null and component=="ivt"): 
                try:
                    n_inst = len(np.unique(self.bank[:, 1])) -1
                except:
                    n_inst = 6
                classwise = classwise[:-n_inst]
            mean      = np.nanmean(classwise)
        return {"AP":classwise, "mAP":mean}     
    
    def compute_video_AP(self, component="ivt", ignore_null=False):
        """
        compute performance video-wise AP
        """           
        global_targets      = self.global_targets
        global_predictions  = self.global_predictions
        if len(self.targets) > 0:
            global_targets.append(self.targets)
            global_predictions.append(self.predictions)
        video_log = []
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='')
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for targets, predicts in zip(global_targets, global_predictions):
                if component in ["ivt", "it", "iv", "t", "v", "i"]:
                    targets  = self.extract(targets, component)
                    predicts = self.extract(predicts, component)
                else:
                    sys.exit("Function filtering {} not yet supported!".format(component))
                
                if targets.size == 0 or predicts.size == 0:
                    
                    print(f"[WARN] compute_video_AP({component}) received empty array during video processing. Skipping this video.")
                    continue  # skip this video

                classwise = average_precision_score(targets, predicts, average=None)
                classwise = self.resolve_nan(classwise)
                video_log.append( classwise.reshape([1,-1]) )
            
            if not video_log:
                 print(f"[ERROR] compute_video_AP({component}) failed to produce a valid video log. Returning mAP 0.0.")
                 return {"AP": np.array([]), "mAP": 0.0}
                 
            video_log = np.concatenate(video_log, axis=0)        
            videowise = np.nanmean(video_log, axis=0)
            if (ignore_null and component=="ivt"): 
                try:
                    n_inst = len(np.unique(self.bank[:, 1])) - 1  # exclude -1 (null)
                except:
                    n_inst = 6  # fallback to 6 on failure
                videowise = videowise[:-n_inst]
            mean      = np.nanmean(videowise)
        return {"AP":videowise, "mAP":mean}
    
    
    def compute_video_AP_only(self, ignore_null: bool = False):
        """
        Video-wise AP/mAP without extract(): (e.g., instruments: C=6).
        """
        global_targets      = self.global_targets
        global_predictions  = self.global_predictions
        if len(self.targets) > 0:
            global_targets.append(self.targets)
            global_predictions.append(self.predictions)

        video_log = []
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='')
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for targets, predicts in zip(global_targets, global_predictions):
                assert targets.shape[1] == self.num_class and predicts.shape[1] == self.num_class, \
                    f"[compute_video_AP_only] expected C={self.num_class}, got {targets.shape[1]}/{predicts.shape[1]}"
                classwise = average_precision_score(targets, predicts, average=None)
                classwise = self.resolve_nan(classwise)
                video_log.append(classwise.reshape(1, -1))

        if not video_log:
             print(f"[ERROR] compute_video_AP_only(C={self.num_class}) failed to produce a valid video log. Returning mAP 0.0.")
             return {"AP": np.array([]), "mAP": 0.0}
             
        video_log = np.concatenate(video_log, axis=0)    # [V, C]
        videowise = np.nanmean(video_log, axis=0)       # (C,)
        mean      = float(np.nanmean(videowise))
        return {"AP": videowise, "mAP": mean}


    def compute_video_all_AP(self, components=["i", "v", "t", "iv", "it", "ivt"], ignore_null=False):
        """
        Compute performance video-wise AP for multiple components.
        """
        global_targets = self.global_targets
        global_predictions = self.global_predictions

        if len(self.targets) > 0:
            global_targets.append(self.targets)
            global_predictions.append(self.predictions)

        results = {}
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='')
            warnings.simplefilter("ignore", category=RuntimeWarning)

            for component in components:
                video_log = []
                
                for targets, predicts in zip(global_targets, global_predictions):
                    if component in ["ivt", "it", "iv", "t", "v", "i"]:
                        targets = self.extract(targets, component)
                        predicts = self.extract(predicts, component)
                    else:
                        sys.exit(f"Function filtering {component} not yet supported!")
                    
                    if targets.size == 0 or predicts.size == 0:
                        continue  # skip this video

                    classwise = average_precision_score(targets, predicts, average=None)
                    classwise = self.resolve_nan(classwise)
                    video_log.append(classwise.reshape([1, -1]))
                
                if not video_log:
                    print(f"[ERROR] compute_video_all_AP({component}) failed to produce a valid video log. Returning mAP 0.0.")
                    mean_ap = 0.0
                    std_ap = 0.0
                else:
                    video_log = np.concatenate(video_log, axis=0)
                    videowise = np.nanmean(video_log, axis=0)
                    if ignore_null and component == "ivt":
                        try:
                            n_inst = len(np.unique(self.bank[:, 1])) -1
                        except:
                            n_inst = 6
                        videowise = videowise[:-n_inst]
                        
                    mean_ap = np.nanmean(videowise)
                    std_ap = np.nanstd(videowise)

                results[component] = {"mean_AP": mean_ap, "std_AP": std_ap}
        return results

    ##%%%%%%%%%%%%%%%%%%% TOP OP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
    
    def topK(self, k=5, component="ivt"):
        """
        compute topK performance for all seen examples after a reset_global()        
        """         
        global_targets      = self.global_targets
        global_predictions  = self.global_predictions
        if len(self.targets) > 0:
            global_targets.append(self.targets)
            global_predictions.append(self.predictions)
        targets  = np.concatenate(global_targets, axis=0)
        predicts = np.concatenate(global_predictions, axis=0)
        if component in ["ivt", "it", "iv", "t", "v", "i"]:
            targets  = self.extract(targets, component)
            predicts = self.extract(predicts, component)
        else:
            sys.exit("Function filtering {} not supported yet!".format(component))
            
        if targets.size == 0 or predicts.size == 0:
             print(f"[WARN] topK({component}) received empty array. Returning 0.0.")
             return 0.0
             
        correct = 0.0
        total   = 0
        for gt, pd in zip(targets, predicts):
            gt_pos  = np.nonzero(gt)[0]
            pd_idx  = (-pd).argsort()[:k]
            correct += len(set(gt_pos).intersection(set(pd_idx)))
            total   += len(gt_pos)
        if total==0: total=1
        return correct/total

    def topClass(self, k=10, component="ivt"):
        """
        compute top K recognize classes for all seen examples after a reset_global()        
        """
        global_targets      = self.global_targets
        global_predictions  = self.global_predictions
        if len(self.targets) > 0:
            global_targets.append(self.targets)
            global_predictions.append(self.predictions)
        targets  = np.concatenate(global_targets, axis=0)
        predicts = np.concatenate(global_predictions, axis=0)
        if component in ["ivt", "it", "iv", "t", "v", "i"]:
            targets  = self.extract(targets, component)
            predicts = self.extract(predicts, component)
        else:
            sys.exit("Function filtering {} not supported yet!".format(component))
            
        if targets.size == 0 or predicts.size == 0:
             print(f"[WARN] topClass({component}) received empty array. Returning {component}.")
             return {}
             
        classwise = average_precision_score(targets, predicts, average=None)
        classwise = self.resolve_nan(classwise)
        pd_idx    = (-classwise).argsort()[:k]
        output    = {x:classwise[x] for x in pd_idx}
        return output