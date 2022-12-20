from det3d.core.bbox import box_torch_ops
from ..registry import DETECTORS
from .base import BaseDetector
from .. import builder
import torch 
from torch import nn 
from tools.utils import collate_intermediate_frame

from models.matching_self import Matching
from det3d.models.utils.finetune_utils import FrozenBatchNorm2d
import numpy as np
from det3d.torchie.trainer import load_checkpoint

@DETECTORS.register_module
class TwoStageDetector(BaseDetector):
    def __init__(
        self,
        first_stage_cfg,
        second_stage_modules,
        feature_head,
        roi_head, 
        NMS_POST_MAXSIZE,
        num_point=1,
        freeze=False,
        freeze_ts=False,
        pretrained=None,
        use_final_feature=False,
        **kwargs
    ):
        super(TwoStageDetector, self).__init__()
        self.superglue_config = kwargs.pop('superglue_config')
        self.single_det = builder.build_detector(first_stage_cfg, **kwargs)
        self.NMS_POST_MAXSIZE = NMS_POST_MAXSIZE

        
        self.bbox_head = self.single_det.bbox_head

        self.second_stage = nn.ModuleList()
        # can be any number of modules 
        # bird eye view, cylindrical view, image, multiple timesteps, etc.. 
        for module in second_stage_modules:
            self.second_stage.append(builder.build_second_stage_module(module))

        self.roi_head = builder.build_roi_head(roi_head)
        
        
        self.init_weights(pretrained=pretrained)
        if freeze:
            print("Freeze First Stage Network")
            # we train the model in two steps 
            self.single_det = self.single_det.freeze()
        if freeze_ts:
            print("Freeze Second Stage CLS and REG Networks")
            self.roi_head = self.roi_head.freeze_cls()
            self.roi_head = self.roi_head.freeze_reg()

        self.num_point = num_point
        self.use_final_feature = use_final_feature
        self.new_roi_input_channels = roi_head.input_channels
        
        
        self.feature_head = builder.build_feature_head_module(feature_head)
        print('Feature Head initialization done!')
        self.matching = Matching(self.superglue_config).train()
        print('AGNN initialization done!')

    def init_weights(self, pretrained=None):
        if pretrained is None:
            return 
        try:
            load_checkpoint(self, pretrained, strict=False)
            print("init weight from {}".format(pretrained))
        except:
            print("no pretrained model at {}".format(pretrained))
    def combine_loss(self, one_stage_loss, roi_loss, tb_dict):
        one_stage_loss['loss'][0] += (roi_loss)

        for i in range(len(one_stage_loss['loss'])):
            one_stage_loss['roi_reg_loss'].append(tb_dict['rcnn_loss_reg'])
            one_stage_loss['roi_cls_loss'].append(tb_dict['rcnn_loss_cls'])

        return one_stage_loss

    def get_box_center(self, boxes):
        # box [List]
        centers = [] 
        for box in boxes:            
            if self.num_point == 1 or len(box['box3d_lidar']) == 0:
                centers.append(box['box3d_lidar'][:, :3])
                
            elif self.num_point == 5:
                center2d = box['box3d_lidar'][:, :2]
                height = box['box3d_lidar'][:, 2:3]
                dim2d = box['box3d_lidar'][:, 3:5]
                rotation_y = box['box3d_lidar'][:, -1]

                corners = box_torch_ops.center_to_corner_box2d(center2d, dim2d, rotation_y)

                front_middle = torch.cat([(corners[:, 0] + corners[:, 1])/2, height], dim=-1)
                back_middle = torch.cat([(corners[:, 2] + corners[:, 3])/2, height], dim=-1)
                left_middle = torch.cat([(corners[:, 0] + corners[:, 3])/2, height], dim=-1)
                right_middle = torch.cat([(corners[:, 1] + corners[:, 2])/2, height], dim=-1) 

                points = torch.cat([box['box3d_lidar'][:, :3], front_middle, back_middle, left_middle, \
                    right_middle], dim=0)

                centers.append(points)
            else:
                raise NotImplementedError()

        return centers

    def reorder_first_stage_pred_and_feature(self, first_pred, example, features):
        batch_size = len(first_pred)
        box_length = first_pred[0]['box3d_lidar'].shape[1] 
        feature_vector_length = sum([feat[0].shape[-1] for feat in features])

        rois = first_pred[0]['box3d_lidar'].new_zeros((batch_size, 
            self.NMS_POST_MAXSIZE, box_length 
        ))
        orig_num_objs = first_pred[0]['box3d_lidar'].new_zeros((batch_size), dtype=torch.int)
        roi_scores = first_pred[0]['scores'].new_zeros((batch_size,
            self.NMS_POST_MAXSIZE
        ))
        roi_labels = first_pred[0]['label_preds'].new_zeros((batch_size,
            self.NMS_POST_MAXSIZE), dtype=torch.long
        )
        roi_features = features[0][0].new_zeros((batch_size, 
            self.NMS_POST_MAXSIZE, feature_vector_length 
        ))

        for i in range(batch_size):
            num_obj = features[0][i].shape[0]
            # basically move rotation to position 6, so now the box is 7 + C . C is 2 for nuscenes to
            # include velocity target

            box_preds = first_pred[i]['box3d_lidar']

            if self.roi_head.code_size == 9:
                # x, y, z, w, l, h, rotation_y, velocity_x, velocity_y
                box_preds = box_preds[:, [0, 1, 2, 3, 4, 5, 8, 6, 7]]

            rois[i, :num_obj] = box_preds
            orig_num_objs[i] = num_obj
            roi_labels[i, :num_obj] = first_pred[i]['label_preds'] + 1
            roi_scores[i, :num_obj] = first_pred[i]['scores']
            roi_features[i, :num_obj] = torch.cat([feat[i] for feat in features], dim=-1)

        example['rois'] = rois 
        example['orig_num_objs'] = orig_num_objs
        example['roi_labels'] = roi_labels 
        example['roi_scores'] = roi_scores  
        example['roi_features'] = roi_features

        example['has_class_labels']= True 

        return example 

    def post_process(self, batch_dict):
        batch_size = batch_dict['batch_size']
        pred_dicts = [] 

        for index in range(batch_size):
            box_preds = batch_dict['batch_box_preds'][index]
            cls_preds = batch_dict['batch_cls_preds'][index]  # this is the predicted iou 
            label_preds = batch_dict['roi_labels'][index]

            if box_preds.shape[-1] == 9:
                # move rotation to the end (the create submission file will take elements from 0:6 and -1) 
                box_preds = box_preds[:, [0, 1, 2, 3, 4, 5, 7, 8, 6]]

            scores = torch.sqrt(torch.sigmoid(cls_preds).reshape(-1) * batch_dict['roi_scores'][index].reshape(-1)) 
            mask = (label_preds != 0).reshape(-1)

            box_preds = box_preds[mask, :]
            scores = scores[mask]
            labels = label_preds[mask]-1

            # currently don't need nms 
            pred_dict = {
                'box3d_lidar': box_preds,
                'scores': scores,
                'label_preds': labels,
                "metadata": batch_dict["metadata"][index]
            }

            pred_dicts.append(pred_dict)

        return pred_dicts 

    def concat_gnn_features_per_match(self, example, t_t1, batch_size):
        """concatenate GNN features with the current examples features from First Stage
        1920 original features from BEV FS sent through FH to get 256d vec. concat this with 2*128d of both `t` from `t_t1` and `t_t2` 

        Args:
            example (_type_): _description_
            t_t1 (_type_): _description_
            batch_size (_type_): _description_
            type (str, optional): _description_. Defaults to 'default'.

        Returns:
            _type_: _description_
        """

        new_gnn_features = example['roi_features'].new_zeros((batch_size, 
            self.NMS_POST_MAXSIZE, example['roi_features'].shape[2]+self.superglue_config['descriptor_dim']
        ))
        for batch in range(batch_size):
            A = t_t1[batch]['desc0'].permute(0,2,1).squeeze(0)
            new_gnn_features[batch, :example['orig_num_objs'][batch]] = torch.cat((example['roi_features'][batch, :example['orig_num_objs'][batch]],
                                                                                    A),
                                                                                    1)
        
        example['roi_features'] = new_gnn_features
        
        return example

    def forward(self, example, return_loss=True, **kwargs):
        out = self.single_det.forward_two_stage(example, 
            return_loss, **kwargs)

        if len(out) == 5:
            one_stage_pred, bev_feature, voxel_feature, final_feature, one_stage_loss = out 
            example['voxel_feature'] = voxel_feature
        elif len(out) == 3:
            one_stage_pred, bev_feature, one_stage_loss = out 
        else:
            raise NotImplementedError

        # N C H W -> N H W C 
        if self.use_final_feature:
            example['bev_feature'] = final_feature.permute(0, 2, 3, 1).contiguous()
        else:
            example['bev_feature'] = bev_feature.permute(0, 2, 3, 1).contiguous()
        
        centers_vehicle_frame = self.get_box_center(one_stage_pred)

        if self.roi_head.code_size == 7 and return_loss is True:
            # drop velocity 
            example['gt_boxes_and_cls'] = example['gt_boxes_and_cls'][:, :, [0, 1, 2, 3, 4, 5, 6, -1]]

        features = [] 

        for module in self.second_stage:
            feature = module.forward(example, centers_vehicle_frame, self.num_point)
            features.append(feature)
            # feature is two level list 
            # first level is number of two stage information streams
            # second level is batch 

        example = self.reorder_first_stage_pred_and_feature(first_pred=one_stage_pred, example=example, features=features)


        example = self.feature_head(example)
        
        example['batch_size'] = len(example['rois'])
        t_t = self.matching([example, None], example['batch_size'])
        example = self.concat_gnn_features_per_match(example, t_t, example['batch_size'])


        # final classification / regression 
        batch_dict = self.roi_head(example, training=return_loss)

        if return_loss:
            roi_loss, tb_dict = self.roi_head.get_loss()

            return self.combine_loss(one_stage_loss, roi_loss, tb_dict)
        else:
            return self.post_process(batch_dict)