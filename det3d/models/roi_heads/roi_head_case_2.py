# ------------------------------------------------------------------------------
# Portions of this code are from
# OpenPCDet (https://github.com/open-mmlab/OpenPCDet)
# Licensed under the Apache License.
# ------------------------------------------------------------------------------

from torch import batch_norm
import torch.nn as nn
import torch 
from .roi_head_template import RoIHeadTemplate

from det3d.core import box_torch_ops

from ..registry import ROI_HEAD

from models.matchingCP import Matching
from tools.utils import find_nearest
import torch

@ROI_HEAD.register_module
class RoIHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, code_size=7, add_box_param=False, test_cfg=None):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.test_cfg = test_cfg 
        self.code_size = code_size
        self.add_box_param = add_box_param

        pre_channel = input_channels

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel+256, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel+256,
            output_channels=code_size,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')
        
        # self.matching = Matching().eval().double()
        

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)
        
    def nearest_features(self, t1, t, t2, t_t1_matching=None, t_t2_matching=None, batch_size=1):
        
        for batch in range(batch_size):
            t_t1_indexes = find_nearest(t['rois'][batch][:,:3], t1['rois'][batch][:,:3])
            t1['rois'][batch] = t1['rois'][batch][t_t1_indexes]
            t1['roi_labels'][batch] = t1['roi_labels'][batch][t_t1_indexes]
            t1['roi_features'][batch] = t1['roi_features'][batch][t_t1_indexes]
            t1['roi_scores'][batch] = t1['roi_scores'][batch][t_t1_indexes]
            # t_t1_matching[batch]['desc0'] = t_t1_matching[batch]['desc0'][t_t1_indexes]
            # t_t1_matching[batch]['matches0'] = t_t1_matching[batch]['matches0'][t_t1_indexes]
            
            t_t2_indexes = find_nearest(t['rois'][batch][:,:3], t2['rois'][batch][:,:3])
            t2['rois'][batch] = t2['rois'][batch][t_t2_indexes]
            t2['roi_labels'][batch] = t2['roi_labels'][batch][t_t2_indexes]
            t2['roi_features'][batch] = t2['roi_features'][batch][t_t2_indexes]
            t2['roi_scores'][batch] = t2['roi_scores'][batch][t_t2_indexes]
            
        return [t1, t, t2]
    
    def combine_roi_features(self, t1, t, t2, batch_size=1):
        combined_roi_features = t.new_zeros((t.shape[0], t.shape[1], t.shape[2]*3))
        for batch in range(batch_size):
            combined_roi_features[batch] = torch.cat((t1[batch], t[batch], t2[batch]), 1)
        return combined_roi_features
    
    def combine_shared_features(self, shared_features, t_t1_matching, t_t2_matching, batch_size=1):
        
        combined_shared_features = shared_features.new_zeros((
                                                                shared_features.shape[0], 
                                                                shared_features.shape[1], 
                                                                shared_features.shape[2]  + t_t1_matching[0]['desc0'].shape[1]*4,
                                                                shared_features.shape[3],
                                                            ))
        for batch in range(batch_size):
            combined_shared_features[batch] = torch.cat((
                                                    shared_features[batch].permute(2,0,1), 
                                                    t_t1_matching[batch]['desc0'].permute(0,2,1), 
                                                    t_t1_matching[batch]['desc1'].permute(0,2,1), 
                                                    t_t2_matching[batch]['desc0'].permute(0,2,1), 
                                                    t_t2_matching[batch]['desc1'].permute(0,2,1)
                                                ), 2).permute(1,2,0)
        
        return combined_shared_features.reshape(-1, combined_shared_features.shape[2], combined_shared_features.shape[3]).float().contiguous()

    def forward(self, batch_dict, frame_history=[], training=True):
        """
        :param input_data: input dict
        :return:
        """
        
        batch_dict['batch_size'] = len(batch_dict['rois'])
        #resort by finindg nearest point for every point in batch_dict
        frame_history[0], batch_dict, frame_history[1] = self.nearest_features(
                                                                                frame_history[0], 
                                                                                batch_dict, 
                                                                                frame_history[1],
                                                                                batch_size=batch_dict['batch_size'])
        
        if training:
            targets_dict, sampled_inds = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            batch_dict['roi_features'] = targets_dict['roi_features']
            batch_dict['roi_scores'] = targets_dict['roi_scores']
            
            frame_history_targets = []
            for frame in frame_history:
                # frame['batch_size'] = len(frame['rois'])
                # frame_targets_dict = self.assign_targets(frame)
                frame['rois'] = frame['rois'][sampled_inds]
                frame['roi_labels'] = frame['roi_labels'][sampled_inds]
                frame['roi_features'] = frame['roi_features'][sampled_inds]
                frame['roi_scores'] = frame['roi_scores'][sampled_inds]
                frame_history_targets.append(frame)
                # del frame_targets_dict
        else: #fix this later
            frame_history_targets=frame_history

        
        
        
        # First did it beofre because after nearest map it messes it up (not sure)
        # but features are not sorted so bad.
        t_t1_matching = self.matching([batch_dict, frame_history_targets[0]], batch_dict['batch_size']) 
        t_t2_matching = self.matching([batch_dict, frame_history_targets[1]], batch_dict['batch_size'])
        
        combined_roi_features = self.combine_roi_features(frame_history_targets[0]['roi_features'], batch_dict['roi_features'], frame_history_targets[1]['roi_features'], batch_dict['batch_size'])
        
        # RoI aware pooling
        # pooled_features = batch_dict['roi_features'].reshape(-1, 1,
        #     batch_dict['roi_features'].shape[-1]).contiguous()  # (BxN, 1, C)
        
        # if self.add_box_param:
        #     batch_dict['roi_features'] = torch.cat([batch_dict['roi_features'], batch_dict['rois'], batch_dict['roi_scores'].unsqueeze(-1)], dim=-1)
        pooled_features = combined_roi_features.reshape(-1, 1,
            combined_roi_features.shape[-1]).contiguous()  # From (B, N, C) to (BxN, 1, C)

        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).contiguous() # (BxN, C, 1)

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        
        shared_features = self.combine_shared_features( shared_features.reshape(batch_dict['batch_size'], -1, shared_features.shape[1], shared_features.shape[2]), # think about contiguos later
                                                        t_t1_matching, 
                                                        t_t2_matching,
                                                        batch_dict['batch_size'])
        
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if not training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict
        
        return batch_dict        