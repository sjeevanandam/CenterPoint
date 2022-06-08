import numpy as np
import torch
import os
import cv2
import math
from datetime import datetime 
from time import time
import pickle

from scipy.spatial.distance import cdist
from torch.utils.data import Dataset

class SparseDataset(Dataset):
    """Sparse correspondences dataset."""

    def __init__(self, train_path, nfeatures):

        self.files = []
        self.files += [train_path + f for f in os.listdir(train_path)]

        self.nfeatures = nfeatures
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.nfeatures)
        self.matcher = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=False)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        file_name = self.files[idx]
        image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE) 
        sift = self.sift
        width, height = image.shape[:2]
        corners = np.array([[0, 0], [0, height], [width, 0], [width, height]], dtype=np.float32)
        warp = np.random.randint(-224, 224, size=(4, 2)).astype(np.float32)

        # get the corresponding warped image
        M = cv2.getPerspectiveTransform(corners, corners + warp)
        warped = cv2.warpPerspective(src=image, M=M, dsize=(image.shape[1], image.shape[0])) # return an image type
        
        # extract keypoints of the image pair using SIFT
        kp1, descs1 = sift.detectAndCompute(image, None)
        kp2, descs2 = sift.detectAndCompute(warped, None)

        # limit the number of keypoints
        kp1_num = min(self.nfeatures, len(kp1))
        kp2_num = min(self.nfeatures, len(kp2))
        kp1 = kp1[:kp1_num]
        kp2 = kp2[:kp2_num]

        kp1_np = np.array([(kp.pt[0], kp.pt[1]) for kp in kp1])
        kp2_np = np.array([(kp.pt[0], kp.pt[1]) for kp in kp2])

        # skip this image pair if no keypoints detected in image
        if len(kp1) < 1 or len(kp2) < 1:
            return{
                'keypoints0': torch.zeros([0, 0, 2], dtype=torch.double),
                'keypoints1': torch.zeros([0, 0, 2], dtype=torch.double),
                'descriptors0': torch.zeros([0, 2], dtype=torch.double),
                'descriptors1': torch.zeros([0, 2], dtype=torch.double),
                'image0': image,
                'image1': warped,
                'file_name': file_name
            } 

        # confidence of each key point
        scores1_np = np.array([kp.response for kp in kp1]) 
        scores2_np = np.array([kp.response for kp in kp2])

        kp1_np = kp1_np[:kp1_num, :]
        kp2_np = kp2_np[:kp2_num, :]
        descs1 = descs1[:kp1_num, :]
        descs2 = descs2[:kp2_num, :]

        # obtain the matching matrix of the image pair
        matched = self.matcher.match(descs1, descs2)
        kp1_projected = cv2.perspectiveTransform(kp1_np.reshape((1, -1, 2)), M)[0, :, :] 
        dists = cdist(kp1_projected, kp2_np)

        min1 = np.argmin(dists, axis=0)
        min2 = np.argmin(dists, axis=1)

        min1v = np.min(dists, axis=1)
        min1f = min2[min1v < 3]

        xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]
        matches = np.intersect1d(min1f, xx)

        missing1 = np.setdiff1d(np.arange(kp1_np.shape[0]), min1[matches])
        missing2 = np.setdiff1d(np.arange(kp2_np.shape[0]), matches)

        MN = np.concatenate([min1[matches][np.newaxis, :], matches[np.newaxis, :]])
        MN2 = np.concatenate([missing1[np.newaxis, :], (len(kp2)) * np.ones((1, len(missing1)), dtype=np.int64)])
        MN3 = np.concatenate([(len(kp1)) * np.ones((1, len(missing2)), dtype=np.int64), missing2[np.newaxis, :]])
        all_matches = np.concatenate([MN, MN2, MN3], axis=1)

        kp1_np = kp1_np.reshape((1, -1, 2))
        kp2_np = kp2_np.reshape((1, -1, 2))
        descs1 = np.transpose(descs1 / 256.)
        descs2 = np.transpose(descs2 / 256.)

        image = torch.from_numpy(image/255.).double()[None].cuda()
        warped = torch.from_numpy(warped/255.).double()[None].cuda()

        
        
        return{
            'keypoints0': list(kp1_np),
            'keypoints1': list(kp2_np),
            'descriptors0': list(descs1),
            'descriptors1': list(descs2),
            'scores0': list(scores1_np),
            'scores1': list(scores2_np),
            'image0': image,
            'image1': warped,
            'all_matches': list(all_matches),
            'file_name': file_name
        } 

class WaymoGTDataset(Dataset):
    """Sparse correspondences dataset."""

    def __init__(self, path, logger):

        self.logger = logger
        self.pairs = []

        
        files = []
        seq_frames = {}
        
        files += self.sort_frames(os.listdir(path))
        for f in files:
            frame_num = '_'.join(f[:-4].split('_')[2:4])
            seq_name = '_'.join(f[:-4].split('_')[:2])
            if seq_name in seq_frames:
                seq_frames[seq_name].append(frame_num)
            else:
                seq_frames[seq_name] = [frame_num]
        for seq, frames in seq_frames.items():
            next_frame_idx = 1
            num_of_frames = len(frames)
            for current_frame_idx in range(num_of_frames):
                if next_frame_idx == num_of_frames:
                    break
                self.pairs.append((path + seq + '_' + frames[current_frame_idx] + '.pkl', path + seq + '_' + frames[next_frame_idx] + '.pkl'))
                next_frame_idx += 1
                

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        start_time = time()
        # self.logger.info('Fetching Item')
        frame_pair = self.pairs[idx]
        
        first_frame = pickle.load(open(frame_pair[0], 'rb'))
        second_frame = pickle.load(open(frame_pair[1], 'rb'))
        # pc = frame['points'][:, :3]
        
        # pc_wraped, R = self.rotate_point_cloud(pc)

        first_frame_box_names = {}
        for f in first_frame['objects']:
            first_frame_box_names[f['name']] = f['box']

        second_frame_box_names = {}
        for f in second_frame['objects']:
            second_frame_box_names[f['name']] = f['box']
        # common_boxes = np.intersect1d(list(first_frame_box_names.keys()), list(second_frame_box_names.keys()))
        
        # boxes1 = []
        # boxes2 = []
        # for common_box in common_boxes:
        #     boxes1.append(first_frame_box_names[common_box])
        #     boxes2.append(second_frame_box_names[common_box])
        boxes1 = np.array(list(first_frame_box_names.values()))
        boxes2 = np.array(list(second_frame_box_names.values()))
        # skip this  pair if no keypoints
        if len(boxes1) <= 1 or len(boxes2) <= 1:
            # self.logger.info("Time elapsed for one item is (hh:mm:ss.ms) {}".format(time()-start_time))
            return{
                'keypoints0': torch.zeros([0, 0, 2], dtype=torch.double),
                'keypoints1': torch.zeros([0, 0, 2], dtype=torch.double),
                'descriptors0': torch.zeros([0, 2], dtype=torch.double),
                'descriptors1': torch.zeros([0, 2], dtype=torch.double),
                'file_name': frame_pair[0]
            }
        
        kps1 = boxes1[:,:3]
        kps2 = boxes2[:,:3]
        
        matches_1_to_2 = np.nonzero(np.in1d(list(first_frame_box_names.keys()), list(second_frame_box_names.keys())))[0]
        matches_2_to_1 = np.nonzero(np.in1d(list(second_frame_box_names.keys()), list(first_frame_box_names.keys())))[0]
        all_matches = np.vstack((matches_1_to_2, matches_2_to_1))
        
        descs1 = np.array(list(first_frame_box_names.values()))[:,3:]
        descs2 = np.array(list(second_frame_box_names.values()))[:,3:]

        # confidence of each key point
        scores_np1 = np.zeros(kps1.shape[0])
        scores_np2 = np.zeros(kps2.shape[0])
        
        scores_np1[matches_1_to_2] = 1.0
        scores_np2[matches_2_to_1] = 1.0
        
        kps1 = kps1.reshape((1, -1, 3))
        kps2 = kps2.reshape((1, -1, 3))
        
        descs1 = descs1.T
        descs2 = descs2.T

        elapsed = time()-start_time
        # self.logger.info('Time elapsed for one LOAD_DATA is %f seconds' %elapsed)
        return{
            'keypoints0': list(kps1),
            'keypoints1': list(kps2),
            'descriptors0': list(descs1),
            'descriptors1': list(descs2),
            'scores0': list(scores_np1),
            'scores1': list(scores_np2),
            # 'image0': image,
            # 'image1': warped,
            'all_matches': list(all_matches),
            'file_name': frame_pair[0][0]
        }

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

    def rotate_point_cloud(self, data):
        """ Randomly rotate the point clouds to augument the dataset
            rotation is per shape based along up direction
            Input:
            Nx3 array, original point clouds
            Return:
            Nx3 array, rotated point clouds
        """
        rotated_data = np.zeros(data.shape, dtype=np.float32)

        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        rotated_data = np.dot(data.reshape((-1, 3)), rotation_matrix)

        return rotated_data, rotation_matrix