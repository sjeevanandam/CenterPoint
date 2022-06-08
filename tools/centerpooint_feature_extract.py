# modified from the single_inference.py by @muzi2045
from spconv.utils import VoxelGenerator as VoxelGenerator
from det3d.datasets.pipelines.loading import read_single_waymo
from det3d.datasets.pipelines.loading import get_obj
from det3d.torchie.trainer import load_checkpoint
from det3d.models import build_detector
from det3d.torchie import Config
from tqdm import tqdm 
import numpy as np
import pickle 
import open3d as o3d
import argparse
import torch
import time 
import os
from scipy.spatial.distance import cdist

from det3d.torchie.apis import get_root_logger
from det3d.torchie import Config

from det3d.core.bbox import box_np_ops

voxel_generator = None 
model = None 
device = None 
activation = {}

def initialize_model(args):
    global model, voxel_generator  
    cfg = Config.fromfile(args.config)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location="cpu")
    # print(model)
    if args.fp16:
        print("cast model to fp16")
        model = model.half()

    model = model.cuda()
    model.eval()

    global device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    range = cfg.voxel_generator.range
    voxel_size = cfg.voxel_generator.voxel_size
    max_points_in_voxel = cfg.voxel_generator.max_points_in_voxel
    max_voxel_num = cfg.voxel_generator.max_voxel_num[1]
    voxel_generator = VoxelGenerator(
        voxel_size=voxel_size,
        point_cloud_range=range,
        max_num_points=max_points_in_voxel,
        max_voxels=max_voxel_num
    )
    return model 

def voxelization(points, voxel_generator):
    voxel_output = voxel_generator.generate(points)  
    voxels, coords, num_points = \
        voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']

    return voxels, coords, num_points  

def _process_inputs(points, fp16):
    voxels, coords, num_points = voxel_generator.generate(points)
    num_voxels = np.array([voxels.shape[0]], dtype=np.int32)
    grid_size = voxel_generator.grid_size
    coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values = 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    voxels = torch.tensor(voxels, dtype=torch.float32, device=device)
    coords = torch.tensor(coords, dtype=torch.int32, device=device)
    num_points = torch.tensor(num_points, dtype=torch.int32, device=device)
    num_voxels = torch.tensor(num_voxels, dtype=torch.int32, device=device)

    if fp16:
        voxels = voxels.half()

    inputs = dict(
            voxels = voxels,
            num_points = num_points,
            num_voxels = num_voxels,
            coordinates = coords,
            shape = [grid_size]
        )

    return inputs 

def run_model(points, fp16=False):
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    # model.bbox_head.tasks[0].reg[1].register_forward_hook(get_activation('reg'))
    # model.reader.register_forward_hook(get_activation('reader'))
    # model.backbone.register_forward_hook(get_activation('backbone'))
    # model.neck.blocks[2].register_forward_hook(get_activation('blocks'))
    # model.neck.register_forward_hook(get_activation('neck'))
    # model.bbox_head.shared_conv.register_forward_hook(get_activation('before_reg'))
    # model.bbox_head.tasks[0].reg.register_forward_hook(get_activation('after_reg'))
    with torch.no_grad():
        data_dict = _process_inputs(points, fp16)
        outputs = model(data_dict, return_loss=False)[0]

    # print(activation['reader'])
    return {'boxes': outputs['box3d_lidar'].cpu().numpy(),
        'scores': outputs['scores'].cpu().numpy(),
        'classes': outputs['label_preds'].cpu().numpy()}, data_dict, activation['reader']


def sort_frames(frames):
    indices = [] 

    for f in frames:
        seq_id = int(f.split("_")[1])
        frame_id= int(f.split("_")[3][:-4])

        idx = seq_id * 1000 + frame_id
        indices.append(idx)

    rank = list(np.argsort(np.array(indices)))

    frames = [frames[r] for r in rank]
    return frames

def process_example(points, fp16=False):
    output, data_dict, inter_features = run_model(points, fp16)

    assert len(output) == 3
    assert set(output.keys()) == set(('boxes', 'scores', 'classes'))
    num_objs = output['boxes'].shape[0]
    assert output['scores'].shape[0] == num_objs
    assert output['classes'].shape[0] == num_objs

    return output, data_dict, inter_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CenterPoint")
    parser.add_argument("config", help="path to config file")
    parser.add_argument(
        "--checkpoint", help="the path to checkpoint which the model read from", default=None, type=str
    )
    parser.add_argument('--input_data_dir', type=str, required=True)
    parser.add_argument('--annos_data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--threshold', default=0.5)
    parser.add_argument('--visual', action='store_true')
    parser.add_argument("--online", action='store_true')
    parser.add_argument('--num_frame', default=-1, type=int)
    args = parser.parse_args()

    print("Please prepare your point cloud in waymo format and save it as a pickle dict with points key into the {}".format(args.input_data_dir))
    print("One point cloud should be saved in one pickle file.")
    print("Download and save the pretrained model at {}".format(args.checkpoint))

    cfg = Config.fromfile(args.config)
    logger = get_root_logger(cfg.log_level)
    # Run any user-specified initialization code for their submission.
    model = initialize_model(args)
    
    latencies = []
    visual_dicts = {}
    pred_dicts = {}
    counter = 0
    
    frames = sort_frames(os.listdir(args.input_data_dir))
    total_frames = len(frames)
    
    for frame_name in tqdm(frames):
        counter += 1 

        frame_num = '_'.join(frame_name[:-4].split('_')[2:4])
        seq_name = '_'.join(frame_name[:-4].split('_')[:2])
        
        logger.info("Processing {}/{}".format(frame_num[-1], total_frames))

        pc_name = os.path.join(args.input_data_dir, frame_name)
        # points = pickle.load(open(pc_name, 'rb'))['points']
        points = read_single_waymo(get_obj(pc_name))

        detections, data_dicts, inter_features = process_example(points, args.fp16)
        
        gt = pickle.load(open(os.path.join(args.annos_data_dir, frame_name), 'rb'))['objects']
        
        gt_boxes = []
        gt_classes = []
        for b in gt:
            gt_boxes.append(b['box'][..., [0, 1, 2, 3, 4, 5, -1]])
            gt_classes.append(b['label'] - 1)
        
        gt_objs = {}
        gt_objs['boxes'] = np.array(gt_boxes, dtype=np.float32)
        if not gt_objs['boxes'].size == 0:
            gt_objs['boxes'][:, -1] = -np.pi / 2 - gt_objs['boxes'][:, -1]
            gt_objs['boxes'][:, [3, 4]] = gt_objs['boxes'][:, [4, 3]]
            
            gt_objs['classes'] = np.array(gt_classes)
            gt_objs['scores'] = np.ones(gt_boxes.__len__())
        start_time = time.time()
        voxels = data_dicts['voxels'].reshape(-1,5).cpu().numpy()
        voxels_flattened = voxels.reshape(-1,5)[:,:3]
        
        keypoint_features = []
        for keypoint in gt_objs['boxes']:
        
            value = keypoint[:3]
            idx = cdist(voxels_flattened, np.atleast_2d(value)).argmin()
            voxel_id = int(np.floor(idx/20))
            keypoint_features.append(inter_features[voxel_id].cpu().numpy())
        gt_objs['centerpoint_features'] = np.array(keypoint_features)
        print(time.time() - start_time)
        # tree = KDTree(data)
        # _, idx = tree.query(value)
        
        if counter%100 == 0:
            print(gt_boxes)
        
        
        # pred_dicts.update({frame_name: detections})

    # if args.visual:
    #     with open(os.path.join(args.output_dir, 'visualization.pkl'), 'wb') as f:
    #         pickle.dump(visual_dicts, f)

    # with open(os.path.join(args.output_dir, 'detections.pkl'), 'wb') as f:
    #     pickle.dump(pred_dicts, f)
