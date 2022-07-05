import sys
import pickle
import json
import random
import operator
from numba.cuda.simulator.api import detect
import numpy as np

from functools import reduce
from pathlib import Path
from copy import deepcopy

from det3d.datasets.custom import PointCloudDataset

from det3d.datasets.registry import DATASETS


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
        print("Using {} sweeps".format(nsweeps))
        super(WaymoDataset, self).__init__(
            root_path, info_path, pipeline, test_mode=test_mode, class_names=class_names
        )

        self._info_path = info_path
        self._class_names = class_names
        self._num_point_features = WaymoDataset.NumPointFeatures if nsweeps == 1 else WaymoDataset.NumPointFeatures+1
        
        self.last_tokens = {}
        for info in self._waymo_infos:
            seq = info['token'].split('.')[0].split('_')[1]
            frame = info['token'].split('.')[0].split('_')[3]
            self.last_tokens.update({seq: frame})

    def reset(self):
        assert False 

    def load_infos(self, info_path):

        with open(self._info_path, "rb") as f:
            _waymo_infos_all = pickle.load(f)

        self._waymo_infos = _waymo_infos_all[::self.load_interval]

        print("Using {} Frames".format(len(self._waymo_infos)))

    def __len__(self):

        if not hasattr(self, "_waymo_infos"):
            self.load_infos(self._info_path)

        return len(self._waymo_infos)
    
    def get_res(self):
        res = {}
        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
                "annotations": None,
                "nsweeps": self.nsweeps, 
            },
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": self._num_point_features,
                "token": "",
            },
            "calib": None,
            "cam": {},
            "mode": "val" if self.test_mode else "train",
            "type": "WaymoDataset",
        }
        return res
    
    def get_sensor_data(self, idx):
        info = self._waymo_infos[idx]
        # print("Current ID IS!! {} and frame is {}".format(idx, info['token']))
        res = self.get_res()
        res["metadata"]["token"] = info["token"]

        data, _ = self.pipeline(res, info)

        seq = info['token'].split('.')[0].split('_')[1]
        frame = info['token'].split('.')[0].split('_')[3]
        if int(frame) == 0:
            # data["frame_history"] = None
            # return data
            #previous frame and next frame
            t1_info = self._waymo_infos[idx+1]
            t2_info = self._waymo_infos[idx+2]
            
            # use transforms on t1 first
            res = self.get_res()
            res["metadata"]["token"] = t1_info["token"] #Just updating token
            t1_data, _ = self.pipeline(res, t1_info) #applying the transform
            # now on t2
            res = self.get_res()
            res["metadata"]["token"] = t2_info["token"] #Just updating token
            t2_data, _ = self.pipeline(res, t2_info) #applying the transform
            
            # data["frame_history"] = {
            #     "t1": t1_data,
            #     "t2": t2_data
            # }
            data["t1"] = t1_data
            data["t2"] = t2_data
            return data
        elif int(frame) == int(self.last_tokens[seq]):
            #previous frame and next frame
            t1_info = self._waymo_infos[idx-1]
            t2_info = self._waymo_infos[idx-2]
            
            # use transforms on t1 first
            res = self.get_res()
            res["metadata"]["token"] = t1_info["token"] #Just updating token
            t1_data, _ = self.pipeline(res, t1_info) #applying the transform
            # now on t2
            res = self.get_res()
            res["metadata"]["token"] = t2_info["token"] #Just updating token
            t2_data, _ = self.pipeline(res, t2_info) #applying the transform
            
            # data["frame_history"] = {
            #     "t1": t1_data,
            #     "t2": t2_data
            # }
            data["t1"] = t1_data
            data["t2"] = t2_data
            return data
        else:
            #previous frame and next frame
            t1_info = self._waymo_infos[idx-1]
            t2_info = self._waymo_infos[idx+1]
            
            # use transforms on t1 first
            res = self.get_res()
            res["metadata"]["token"] = t1_info["token"] #Just updating token
            t1_data, _ = self.pipeline(res, t1_info) #applying the transform
            # now on t2
            res = self.get_res()
            res["metadata"]["token"] = t2_info["token"] #Just updating token
            t2_data, _ = self.pipeline(res, t2_info) #applying the transform
            
            # data["frame_history"] = {
            #     "t1": t1_data,
            #     "t2": t2_data
            # }
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

