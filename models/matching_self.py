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
from .superglue_self import SuperGlue
from torch.autograd import Variable
from det3d.models.utils.finetune_utils import FrozenBatchNorm2d
import numpy as np
import shutil

class Matching(torch.nn.Module):

    def __init__(self, superglue_config):
        super().__init__()
        
        shutil.copy("./models/superglueGNN.py", superglue_config['work_dir'])
        config = {
                'weights': 'indoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
                'checkpoint': "/netscratch/jeevanandam/thesis/SuperGlue_Waymo/work_dirs/with_no_features_75_train_25_val/model_epoch_20.pth"
        }
        self.superglue = SuperGlue({**config, **superglue_config})
        self.desc_dim = self.superglue.config['descriptor_dim']
        
        
        

    def forward(self, data, batch_size=1, out_type='roi'):
        """ Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        
        all_preds = []
        
        for batch in range(batch_size):
            pred = {}
            
            boxes1 = data[0]['rois'][batch, :data[0]['orig_num_objs'][batch]]

            
            if len(boxes1) <= 1:
                # self.logger.info("Time elapsed for one item is (hh:mm:ss.ms) {}".format(time()-start_time))
                pred = {
                    'keypoints0': torch.zeros([0, 0, 2]),

                    'descriptors0': torch.zeros([0, 2]),

                    'file_name': "frame_pair[0]"
                }
            else:
            
                kps1 = boxes1[:,:3]

                
                kps1 = kps1.reshape((1, -1, 3))
                

                scores1 = data[0]['roi_scores'][batch, :data[0]['orig_num_objs'][batch]].unsqueeze(1)

                
                
                
                if out_type == 'roi':
                    descriptors1 = data[0]['roi_features'][batch, :data[0]['orig_num_objs'][batch]]
                    # descriptors2 = data[1]['roi_features'][batch, :data[1]['orig_num_objs'][batch]]
                    
                    pred = {
                        'keypoints0': list(kps1),
                        'descriptors0': list(descriptors1),
                        'scores0': list(scores1),
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
    
    
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self