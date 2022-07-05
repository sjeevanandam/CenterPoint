import sys
import pickle
import json
import random
import operator
from numba.cuda.simulator.api import detect
import numpy as np
import os
import torch
import io

from functools import reduce
from pathlib import Path
from copy import deepcopy

from det3d.datasets.custom import PointCloudDataset

from det3d.datasets.registry import DATASETS

## THIS IS FS Features extracted DATASET VERSION

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

@DATASETS.register_module
class WaymoDataset(PointCloudDataset):
    NumPointFeatures = 5  # x, y, z, intensity, elongation

    def __init__(
        self,
        info_path,
        root_path,
        cfg=None,
        pipeline=None,
        class_names=None,
        test_mode=False,
        sample=False,
        nsweeps=1,
        load_interval=1,
        **kwargs,
    ):
        self.load_interval = load_interval 
        self.sample = sample
        self.nsweeps = nsweeps
        self.fs_features_path = root_path+"/fs_features_new/"
        print("Using {} sweeps".format(nsweeps))
        super(WaymoDataset, self).__init__(
            root_path, info_path, pipeline, test_mode=test_mode, class_names=class_names
        )

        self._info_path = info_path
        self._class_names = class_names
        self._num_point_features = WaymoDataset.NumPointFeatures if nsweeps == 1 else WaymoDataset.NumPointFeatures+1
        
        self.last_tokens = {}
        for info in self._waymo_infos:
            seq = info.split('.')[0].split('_')[1]
            frame = info.split('.')[0].split('_')[3]
            self.last_tokens.update({seq: frame})

    def reset(self):
        assert False 

    def load_infos(self, info_path):

        all_frames = []
        
        all_frames += self.sort_frames(os.listdir(info_path))

        self._waymo_infos = all_frames[::self.load_interval]

        print("Using {} Frames".format(len(self._waymo_infos)))

    def __len__(self):

        if not hasattr(self, "_waymo_infos"):
            self.load_infos(self.fs_features_path)

        return len(self._waymo_infos)

    def get_sensor_data(self, idx):
        info = self._waymo_infos[idx]
        
        with open(self.fs_features_path+info, "rb") as f:
            data = CPU_Unpickler(f).load()

        seq = info.split('.')[0].split('_')[1]
        frame = info.split('.')[0].split('_')[3]
        if int(frame) == 0:
            # data["frame_history"] = None
            # return data
            #previous frame and next frame
            t1_info = self._waymo_infos[idx+1]
            t2_info = self._waymo_infos[idx+2]
            
            with open(self.fs_features_path+t1_info, "rb") as f:
                t1_data = CPU_Unpickler(f).load()
                
            with open(self.fs_features_path+t2_info, "rb") as f:
                t2_data = CPU_Unpickler(f).load()
            data["t1"] = t1_data
            data["t2"] = t2_data
            return data
        elif int(frame) == int(self.last_tokens[seq]):
            #previous frame and next frame
            t1_info = self._waymo_infos[idx-1]
            t2_info = self._waymo_infos[idx-2]
            
            with open(self.fs_features_path+t1_info, "rb") as f:
                t1_data = CPU_Unpickler(f).load()
                
            with open(self.fs_features_path+t2_info, "rb") as f:
                t2_data = CPU_Unpickler(f).load()
            data["t1"] = t1_data
            data["t2"] = t2_data
            return data
        else:
            #previous frame and next frame
            t1_info = self._waymo_infos[idx-1]
            t2_info = self._waymo_infos[idx+1]
            
            with open(self.fs_features_path+t1_info, "rb") as f:
                t1_data = CPU_Unpickler(f).load()
                
            with open(self.fs_features_path+t2_info, "rb") as f:
                t2_data = CPU_Unpickler(f).load()
            data["t1"] = t1_data
            data["t2"] = t2_data
            return data

    def __getitem__(self, idx):
        return self.get_sensor_data(idx)

    def evaluation(self, detections, output_dir=None, testset=False):
        from .waymo_common import _create_pd_detection, reorganize_info

        infos = self._waymo_infos 
        infos = reorganize_info(infos)

        _create_pd_detection(detections, infos, output_dir)

        print("use waymo devkit tool for evaluation")

        return None, None 
    
    def sort_frames(self, frames):
        indices = [] 

        for f in frames:
            seq_id = int(f.split("_")[1])
            frame_id= int(f.split("_")[3][:-4])

            idx = seq_id * 1000 + frame_id
            indices.append(idx)

        rank = list(np.argsort(np.array(indices)))

        frames = [frames[r] for r in rank]
        return frames