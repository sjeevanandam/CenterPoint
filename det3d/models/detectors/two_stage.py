from det3d.core.bbox import box_torch_ops
from ..registry import DETECTORS
from .base import BaseDetector
from .. import builder
import torch 
from torch import nn 
from tools.utils import collate_intermediate_frame
import pickle
from models.matchingCP import Matching

# this for FS feature extraction
@DETECTORS.register_module
class TwoStageDetector(BaseDetector):
    def __init__(
        self,
        first_stage_cfg,
        second_stage_modules,
        roi_head, 
        NMS_POST_MAXSIZE,
        num_point=1,
        freeze=False,
        use_final_feature=False,
        **kwargs
    ):
        super(TwoStageDetector, self).__init__()
        self.single_det = builder.build_detector(first_stage_cfg, **kwargs)
        self.NMS_POST_MAXSIZE = NMS_POST_MAXSIZE

        if freeze:
            print("Freeze First Stage Network")
            # we train the model in two steps 
            self.single_det = self.single_det.freeze()
        self.bbox_head = self.single_det.bbox_head

        self.second_stage = nn.ModuleList()
        # can be any number of modules 
        # bird eye view, cylindrical view, image, multiple timesteps, etc.. 
        for module in second_stage_modules:
            self.second_stage.append(builder.build_second_stage_module(module))

        self.roi_head = builder.build_roi_head(roi_head)

        self.num_point = num_point
        self.use_final_feature = use_final_feature
        
        self.matching = Matching().eval().double()

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
            roi_labels[i, :num_obj] = first_pred[i]['label_preds'] + 1
            roi_scores[i, :num_obj] = first_pred[i]['scores']
            roi_features[i, :num_obj] = torch.cat([feat[i] for feat in features], dim=-1)

        example['rois'] = rois 
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
    
    def combine_match_features(self, frame1, frame2, match_dict, batch_size):
        
        assert frame1['roi_features'].shape[0] == match_dict.shape[0]
        
        for batch in range(batch_size):
            matches = match_dict[batch]['matches0']
            valid = (match_dict[batch]['matches0'] > -1).nonzero().squeeze()
            frame1['roi_features'][batch][valid] = torch.mean( torch.stack((frame1['roi_features'][batch][valid], 
                                                                            frame2['roi_features'][batch][matches[valid]])), 
                                                                dim=0)
            
        
        return frame1

    def loss_to_cpu(self, losses):
        for k,v in losses.items():
            if k in ["loss", "loc_loss", "num_positive"]:
                losses[k] = v[0].cpu()
        return losses
    
    def match_to_cpu(self, matches):
        for k,v in matches.items():
            matches[k] = v[0].detach().cpu()
        return matches
    
    def forward(self, example, return_loss=True, **kwargs):
        out = self.single_det.forward_two_stage(example, 
            return_loss, **kwargs)
        t1_data, t2_data = collate_intermediate_frame([example['t1'], example['t2']])
        
        t1_out = self.single_det.forward_two_stage(t1_data, 
            return_loss=False, **kwargs)
        t2_out = self.single_det.forward_two_stage(t2_data, 
            return_loss=False, **kwargs)
        
        del t1_data, t2_data
        
        example['batch_size'] = len(example['metadata'])
        t_t1_matching = self.matching([out, t1_out], example['batch_size'], out_type='lidar') # doing it before because after nearest map it messes it up
        t_t2_matching = self.matching([out, t2_out], example['batch_size'], out_type='lidar')
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
            # centers_vehicle_frame used just to compute the feature vector from bev map for the center
            feature = module.forward(example, centers_vehicle_frame, self.num_point)
            features.append(feature)
            # feature is two level list 
            # first level is number of two stage information streams
            # second level is batch 

        example = self.reorder_first_stage_pred_and_feature(first_pred=one_stage_pred, example=example, features=features)
        
        # combining features in two_stage itself so commenting
        # frame_history = [t1_data, t2_data]
        
        # example['batch_size'] = len(example['rois'])
        # t_t1_matching_roi = self.matching([example, frame_history[0]], example['batch_size']) # doing it beofre because after nearest map it messes it up
        # t_t2_matching = self.matching([example, frame_history[1]], example['batch_size'])
        
        # final classification / regression 
        # batch_dict = self.roi_head(example, training=return_loss)
        
        example['batch_size'] = len(example['metadata'])
        
        del example['points']
        del example['voxels']
        del example['hm']
        del example['ind']
        del example['mask']
        del example['cat']
        del example['shape']
        del example['num_points']
        del example['num_voxels']
        del example['coordinates']
        del example['anno_box']
        del example['bev_feature']
        for batch in range(example['batch_size']):
            cur_frame = dict()
            cur_frame.update({"pred_len": one_stage_pred[batch]['box3d_lidar'].__len__()})
            cur_frame.update({"gt_boxes_and_cls": example['gt_boxes_and_cls'][batch:batch+1].cpu().squeeze()})
            cur_frame.update({"rois": example['rois'][batch:batch+1].cpu().squeeze()})
            cur_frame.update({"roi_labels": example['roi_labels'][batch:batch+1].cpu().squeeze()})
            cur_frame.update({"roi_scores": example['roi_scores'][batch:batch+1].cpu().squeeze()})
            cur_frame.update({"roi_features": example['roi_features'][batch:batch+1].cpu().squeeze()})
            cur_frame.update({'has_class_labels': True})
            cur_frame.update({"metadata": example["metadata"][batch]})
            cur_frame.update({'t_t1_matching':self.match_to_cpu(t_t1_matching[batch])})
            cur_frame.update({'t_t2_matching':self.match_to_cpu(t_t2_matching[batch])})
            cur_frame.update({"one_stage_loss": self.loss_to_cpu(one_stage_loss.copy())})
            with open(str(example['metadata'][batch]['image_prefix']) +'/fs_features/' + example['metadata'][batch]['token'], 'wb') as f:
                pickle.dump(cur_frame, f)
            del cur_frame
        if return_loss:
            # roi_loss, tb_dict = self.roi_head.get_loss()

            return one_stage_loss
        else:
            return self.post_process(batch_dict)
