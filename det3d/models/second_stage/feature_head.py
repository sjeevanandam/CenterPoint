# ------------------------------------------------------------------------------
# Portions of this code are from
# OpenPCDet (https://github.com/open-mmlab/OpenPCDet)
# Licensed under the Apache License.
# ------------------------------------------------------------------------------

from torch import batch_norm
import torch.nn as nn
import torch 

from det3d.core import box_torch_ops

from ..registry import FEATURE_HEAD

@FEATURE_HEAD.register_module
class FeatureHead(nn.Module):
    def __init__(self, input_channels, model_cfg, num_class=1, code_size=7, test_cfg=None):
        super().__init__()
        self.model_cfg = model_cfg
        self.code_size = code_size

        pre_channel = input_channels

        fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.feature_head_layer = nn.Sequential(*fc_list)

        self.init_weights(weight_init='xavier')

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

    def forward(self, batch_dict, training=True):
        """
        :param input_data: input dict
        :return:
        """
        batch_dict['batch_size'] = len(batch_dict['rois'])
        
        pooled_features = batch_dict['roi_features'].reshape(-1, 1,
            batch_dict['roi_features'].shape[-1]).contiguous()  # (BxN, 1, C)

        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).contiguous() # (BxN, C, 1)

        shared_features = self.feature_head_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        shared_features = shared_features.reshape(batch_dict['batch_size'], -1, shared_features.shape[1], shared_features.shape[2]).contiguous().squeeze(dim=-1)
        
        batch_dict['roi_features'] = shared_features
        
        # rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        # rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        # if not training:
        #     batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
        #         batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
        #     )
        #     batch_dict['batch_cls_preds'] = batch_cls_preds
        #     batch_dict['batch_box_preds'] = batch_box_preds
        #     batch_dict['cls_preds_normalized'] = False
        # else:
        #     targets_dict['rcnn_cls'] = rcnn_cls
        #     targets_dict['rcnn_reg'] = rcnn_reg

        #     self.forward_ret_dict = targets_dict
        
        return batch_dict        