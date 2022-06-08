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

from det3d.torchie.apis import get_root_logger
from det3d.torchie import Config

from det3d.core.bbox import box_np_ops

voxel_generator = None 
model = None 
device = None 

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
    with torch.no_grad():
        data_dict = _process_inputs(points, fp16)
        outputs = model(data_dict, return_loss=False)[0]

    return {'boxes': outputs['box3d_lidar'].cpu().numpy(),
        'scores': outputs['scores'].cpu().numpy(),
        'classes': outputs['label_preds'].cpu().numpy()}

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
    output = run_model(points, fp16)

    assert len(output) == 3
    assert set(output.keys()) == set(('boxes', 'scores', 'classes'))
    num_objs = output['boxes'].shape[0]
    assert output['scores'].shape[0] == num_objs
    assert output['classes'].shape[0] == num_objs

    return output    


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

    pairs = []
    files = []
    seq_frames = {}
    max_sequences_train = 30
    max_sequences_val = 10
    
    files += sort_frames(os.listdir(path))
    for f in files:
        frame_num = '_'.join(f[:-4].split('_')[2:4])
        seq_name = '_'.join(f[:-4].split('_')[:2])
        if seq_name in seq_frames:
            seq_frames[seq_name].append(frame_num)
        else:
            seq_frames[seq_name] = [frame_num]

    total_sequences = len(seq_frames)
    processed_sequences = 0
    selected_sequences = 0
    
    processed_sequences_names = []
    for seq, frames in seq_frames.items():
        if is_val:
            if processed_sequences % 2 == 0:
                processed_sequences += 1
                selected_sequences += 1
            else:
                processed_sequences += 1
                continue
            if processed_sequences > total_sequences or selected_sequences >= max_sequences_val+1:
                break
        else:
            if processed_sequences % 2 != 0:
                processed_sequences += 1
                selected_sequences += 1
            else:
                processed_sequences += 1
                continue
            if processed_sequences > total_sequences or selected_sequences >= max_sequences_train+1:
                break
        next_frame_idx = 1
        num_of_frames = len(frames)
        processed_sequences_names.append(seq)
        for current_frame_idx in range(num_of_frames):
            if next_frame_idx == num_of_frames:
                break
            pairs.append((path + seq + '_' + frames[current_frame_idx] + '.pkl', path + seq + '_' + frames[next_frame_idx] + '.pkl'))
            next_frame_idx += 1
    
    
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
        
        logger.info("Processing {}: {}".format(seq_name, frame_num))

        pc_name = os.path.join(args.input_data_dir, frame_name)
        # points = pickle.load(open(pc_name, 'rb'))['points']
        points = read_single_waymo(get_obj(pc_name))

        detections = process_example(points, args.fp16)
        gt = pickle.load(open(os.path.join(args.annos_data_dir, frame_name), 'rb'))['objects']
        
        gt_boxes = []
        gt_classes = []
        for b in gt:
            gt_boxes.append(b['box'][..., [0, 1, 2, 3, 4, 5, -1]])
            gt_classes.append(b['label'] - 1)
        
        gt_objs = {}
        gt_objs['boxes'] = np.array(gt_boxes, dtype=np.float32)
        gt_objs['classes'] = np.array(gt_classes, dtype=np.float32)
        gt_objs['scores'] = np.ones(gt_boxes.__len__(), dtype=np.float32)
        if not gt_objs['boxes'].size == 0:
            gt_objs['boxes'][:, -1] = -np.pi / 2 - gt_objs['boxes'][:, -1]
            gt_objs['boxes'][:, [3, 4]] = gt_objs['boxes'][:, [4, 3]]
            
            gt_objs['classes'] = np.array(gt_classes)
            gt_objs['scores'] = np.ones(gt_boxes.__len__())


        # points_mask = np.zeros([points.shape[0]], bool)
        # points_mask = np.vstack([points_mask]*gt_objs['boxes'].shape[0])
        # fill_colors = np.broadcast_to(np.array([0, 0, 255]), points[:,:3].shape)
        # colors = np.asarray(np.copy(fill_colors))
        # masks = box_np_ops.points_in_rbbox(points, gt_objs['boxes']).T
        # points_mask |= masks
        # points_mask = np.bitwise_or.reduce(points_mask, 0)
        # colors[points_mask == True] = [255, 0, 0]
        # gt_objs['colors'] = colors

        if args.visual and args.online:
            pcd = o3d.geometry.PointCloud()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])

            visual = [pcd]
            num_dets = detections['scores'].shape[0]
            visual += plot_boxes(detections, args.threshold)

            o3d.visualization.draw_geometries(visual)
        elif args.visual:
            # visual_dicts.update({frame_num:{'points': points, 'detections': detections, 'detections_gt': gt_objs}})

            # if (counter == total_frames or '_'.join(frames[counter][:-4].split('_')[:2]) != seq_name):
                
            logger.info("Saving {}: {}".format(seq_name, frame_num))
            with open(os.path.join(args.output_dir, frame_name), 'wb') as f:
                pickle.dump({'points': points, 'detections': detections, 'detections_gt': gt_objs}, f)
            visual_dicts = {}
        
        
        
        
        # pred_dicts.update({frame_name: detections})

    # if args.visual:
    #     with open(os.path.join(args.output_dir, 'visualization.pkl'), 'wb') as f:
    #         pickle.dump(visual_dicts, f)

    # with open(os.path.join(args.output_dir, 'detections.pkl'), 'wb') as f:
    #     pickle.dump(pred_dicts, f)
