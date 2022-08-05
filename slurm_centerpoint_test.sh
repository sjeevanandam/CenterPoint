#!/bin/bash

# Make sure the current working directory is in the python path.
export PYTHONPATH=$PYTHONPATH:$(pwd)
#export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
echo -e "Started  Script \n"

python -m torch.distributed.launch --nproc_per_node=2 ./tools/dist_test.py \
    /netscratch/jeevanandam/thesis/CenterPoint_results/work_dirs/mini/waymo_centerpoint_pp_two_pfn_stride1_two_stage_bev_6epoch_baseline/waymo_centerpoint_pp_two_pfn_stride1_two_stage_bev_6epoch.py \
    --work_dir /netscratch/jeevanandam/thesis/CenterPoint_results/work_dirs/mini/waymo_centerpoint_pp_two_pfn_stride1_two_stage_bev_6epoch_baseline \
    --checkpoint /netscratch/jeevanandam/thesis/CenterPoint_results/work_dirs/mini/waymo_centerpoint_pp_two_pfn_stride1_two_stage_bev_6epoch_baseline/latest.pth

echo -e "\n\n\n Finished Testing Script \n"

