# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

import torch
from .superglueCP import SuperGlue
from torch.autograd import Variable

import numpy as np

class Matching(torch.nn.Module):

    def __init__(self, config={}):
        super().__init__()
        config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': 1024
            },
            'superglue': {
                'weights': 'indoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
                'checkpoint': "/netscratch/jeevanandam/thesis/SuperGlue_Waymo/work_dirs/with_no_features_75_train_25_val/model_epoch_20.pth"
            }
        }
        self.superglue = SuperGlue(config.get('superglue', {}))
        self.desc_dim = self.superglue.config['descriptor_dim']
        
        trained = torch.load(config['superglue']['checkpoint'])
        self.superglue.load_state_dict(trained)
        print("Checkpoint {} with length {} loaded!!!".format(config['superglue']['checkpoint'], trained.__len__()))
        
        
        

    def forward(self, data, batch_size=1, out_type='roi'):
        """ Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        
        all_preds = []
        
        for batch in range(batch_size):
            pred = {}
            
            if out_type == 'roi':
                boxes1 = data[0]['rois'][batch]
                boxes2 = data[1]['rois'][batch]
            elif out_type == 'lidar':
                boxes1 = data[0][0][batch]['box3d_lidar']
                boxes2 = data[1][0][batch]['box3d_lidar']
            
            if len(boxes1) <= 1 or len(boxes2) <= 1:
                # self.logger.info("Time elapsed for one item is (hh:mm:ss.ms) {}".format(time()-start_time))
                pred = {
                    'keypoints0': torch.zeros([0, 0, 2], dtype=torch.double),
                    'keypoints1': torch.zeros([0, 0, 2], dtype=torch.double),
                    'descriptors0': torch.zeros([0, 2], dtype=torch.double),
                    'descriptors1': torch.zeros([0, 2], dtype=torch.double),
                    'file_name': "frame_pair[0]"
                }
            else:
            
                kps1 = boxes1[:,:3]
                kps2 = boxes2[:,:3]

                
                kps1 = kps1.reshape((1, -1, 3))
                kps2 = kps2.reshape((1, -1, 3))
                
                if out_type == 'roi':
                    scores1 = data[0]['roi_scores'][0].unsqueeze(1)
                    scores2 = data[1]['roi_scores'][0].unsqueeze(1)
                elif out_type == 'lidar':
                    scores1 = data[0][0][batch]['scores'].unsqueeze(1)
                    scores2 = data[1][0][batch]['scores'].unsqueeze(1)
                
                pred = {
                    'keypoints0': list(kps1),
                    'keypoints1': list(kps2),
                    'scores0': list(scores1),
                    'scores1': list(scores2),
                    # 'image0': image,
                    # 'image1': warped,
                    'file_name': "frame_pair[0][0]"
                }

            # Batch all features
            # We should either have i) one image per batch, or
            # ii) the same number of local features for all images in the batch.
            # data = {**data, **pred}

            for k in pred:
                if k != 'file_name' and k != 'image0' and k != 'image1' and k != 'test':
                    if type(pred[k]) == torch.Tensor:
                        pred[k] = Variable(pred[k].cuda())
                    else:
                        pred[k] = Variable(torch.stack(pred[k]).cuda())

            # Perform the matching
            pred = {**self.superglue(pred)}
            all_preds.append(pred)
        return np.array(all_preds)
