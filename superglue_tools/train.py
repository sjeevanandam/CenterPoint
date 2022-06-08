from pathlib import Path
import argparse
import numpy as np
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torch.autograd import Variable
from load_data import SparseDataset, WaymoGTDataset
import os
import torch.multiprocessing
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm
from time import time

from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, read_image_modified,
                          get_dist_info, get_root_logger)

from models.superpoint import SuperPoint
from models.superglue import SuperGlue
from models.matchingForTraining import MatchingForTraining

torch.set_grad_enabled(True)
torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():

    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--viz', action='store_true',
        help='Visualize the matches and dump the plots')
    parser.add_argument(
        '--eval', action='store_true',
        help='Perform the evaluation'
                ' (requires ground truth pose and intrinsics)')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by Superpoint'
                ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
                'resize to the exact dimensions, if one number, resize the max '
                'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')

    parser.add_argument(
        '--cache', action='store_true',
        help='Skip the pair if output .npz files are already found')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Plot the keypoints in addition to the matches')
    parser.add_argument(
        '--fast_viz', action='store_true',
        help='Use faster image visualization based on OpenCV instead of Matplotlib')
    parser.add_argument(
        '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
        help='Visualization file extension. Use pdf for highest-quality.')

    parser.add_argument(
        '--opencv_display', action='store_true',
        help='Visualize via OpenCV before saving output images')
    parser.add_argument(
        '--eval_pairs_list', type=str, default='assets/scannet_sample_pairs_with_gt.txt',
        help='Path to the list of image pairs for evaluation')
    parser.add_argument(
        '--shuffle', action='store_true',
        help='Shuffle ordering of pairs before processing')
    parser.add_argument(
        '--max_length', type=int, default=-1,
        help='Maximum number of pairs to evaluate')

    parser.add_argument(
        '--eval_input_dir', type=str, default='assets/scannet_sample_images/',
        help='Path to the directory that contains the images')
    parser.add_argument(
        '--eval_output_dir', type=str, default='dump_match_pairs/',
        help='Path to the directory in which the .npz results and optional,'
                'visualizations are written')
    parser.add_argument(
        '--learning_rate', type=int, default=0.0001,
        help='Learning rate')

    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='batch_size')
    parser.add_argument(
        '--train_path', type=str, default='/home/yinxinjia/yingxin/dataset/COCO2014_train/', 
        help='Path to the directory of training imgs.')
    parser.add_argument(
        '--epoch', type=int, default=20,
        help='Number of epoches')
    parser.add_argument(
        '--mode', type=str, default='default', 
        help='Path to the directory of training imgs.')
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args



def main():
    log_config = dict(
        interval=5,
        hooks=[
            dict(type="TextLoggerHook"),
            # dict(type='TensorboardLoggerHook')
            ],
    )
    log_level = "INFO"
    logger = get_root_logger(log_level)

    opt = parse_args()
    logger.info(opt)

    opt.distributed = True
    # logger.info("DISTRIBUTED WORLD SIZEEE {}".format(os.environ['WORLD_SIZE']))
    # if 'WORLD_SIZE' in os.environ:
    #     opt.distributed = int(os.environ['WORLD_SIZE']) > 1

    if opt.distributed:
        # FOR DISTRIBUTED:  Set the device according to local_rank.
        torch.cuda.set_device(opt.local_rank)
        # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
        # environment variables, and requires that you use init_method=`env://`.
        torch.distributed.init_process_group(backend='nccl',
                                            init_method='env://')
        opt.gpus = torch.distributed.get_world_size()

    # make sure the flags are properly used
    assert not (opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
    assert not (opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (opt.fast_viz and not opt.viz), 'Must use --viz with --fast_viz'
    assert not (opt.fast_viz and opt.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'

    # store viz results
    eval_output_dir = Path(opt.eval_output_dir)
    eval_output_dir.mkdir(exist_ok=True, parents=True)
    logger.info('Will write visualization images to directory "{}"'.format(eval_output_dir))
    
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    

    # load training data
    # train_set = SparseDataset(opt.train_path, opt.max_keypoints)
    if opt.mode == "glue":
        train_set = WaymoGTDataset(opt.train_path, logger)
    else:
        train_set = SparseDataset(opt.train_path, opt.max_keypoints)

    sampler = DistributedSampler(train_set)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=False, batch_size=opt.batch_size, drop_last=True, sampler=sampler, pin_memory=False)

    superglue = SuperGlue(config.get('superglue', {}))

    if torch.cuda.is_available():
        superglue.cuda() # make sure it trains on GPU
    else:
        logger.info("### CUDA not available ###")
    optimizer = torch.optim.Adam(superglue.parameters(), lr=opt.learning_rate)
    mean_loss = []

    logger.info("SuperGlue Model Initialized")
    if opt.distributed == True:
        logger.info("Distributed training enabled.")
        superglue = DDP(superglue,
                        device_ids=[opt.local_rank],
                        output_device=opt.local_rank)
    # start training
    for epoch in range(1, opt.epoch+1):
        logger.info("Epoch [{}/{}] Started.".format(epoch, opt.epoch))
        model_out_path = "/netscratch/jeevanandam/thesis/SuperGlue_Waymo/work_dirs/model_epoch_{}.pth".format(epoch)
        epoch_loss = 0
        superglue.double().train()
        for i, pred in enumerate(tqdm(train_loader)):
            start_time = time()
            for k in pred:
                if k != 'file_name' and k != 'image0' and k != 'image1':
                    if type(pred[k]) == torch.Tensor:
                        pred[k] = Variable(pred[k].cuda())
                    else:
                        pred[k] = Variable(torch.stack(pred[k]).cuda())
                
            data = superglue(pred)
            for k, v in pred.items():
                pred[k] = v[0]
            pred = {**pred, **data}

            if pred['skip_train'] == True: # image has no keypoint
                continue
            
            # process loss
            Loss = pred['loss']
            epoch_loss += Loss.item()
            mean_loss.append(Loss)

            superglue.zero_grad()
            Loss.backward()
            optimizer.step()

            # for every 50 images, print progress and visualize the matches
            if (i+1) % 50 == 0 and opt.mode != "glue":
                logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch, opt.epoch, i+1, len(train_loader), torch.mean(torch.stack(mean_loss)).item())) 
                mean_loss = []

                ### eval ###
                # Visualize the matches.
                superglue.eval()
                # image0, image1 = pred['image0'].cpu().numpy()[0]*255., pred['image1'].cpu().numpy()[0]*255.
                kpts0, kpts1 = pred['keypoints0'].cpu().numpy()[0], pred['keypoints1'].cpu().numpy()[0]
                matches, conf = pred['matches0'].cpu().detach().numpy(), pred['matching_scores0'].cpu().detach().numpy()
                # image0 = read_image_modified(image0, opt.resize, opt.resize_float)
                # image1 = read_image_modified(image1, opt.resize, opt.resize_float)
                valid = matches > -1
                mkpts0 = kpts0[valid]
                mkpts1 = kpts1[matches[valid]]
                mconf = conf[valid]
                # viz_path = eval_output_dir / '{}_matches.{}'.format(str(i), opt.viz_extension)
                # color = cm.jet(mconf)
                stem = pred['file_name']
                text = []

                # make_matching_plot(
                #     image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                #     text, viz_path, stem, stem, opt.show_keypoints,
                #     opt.fast_viz, opt.opencv_display, 'Matches')

            # process checkpoint for every 5e3 images
            if (i+1) % 5e3 == 0:
                logger.info('Epoch [{}/{}], Step [{}/{}], Checkpoint saved to {}'
                    .format(epoch, opt.epoch, i+1, len(train_loader), model_out_path))
            elapsed = time()-start_time
            # logger.info('Time elapsed for one item in TRAINING is {} seconds'.format(elapsed))

        # save checkpoint when an epoch finishes
        epoch_loss /= len(train_loader)
        
        torch.save(superglue, model_out_path)
        logger.info("Epoch [{}/{}] done. Epoch Loss {}. Checkpoint saved to {}"
            .format(epoch, opt.epoch, epoch_loss, model_out_path))
        

if __name__ == '__main__':
    main()