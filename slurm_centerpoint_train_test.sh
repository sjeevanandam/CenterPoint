#!/bin/bash

#to run parallel experiments make a copy everytime
work_dir="/workspace/CenterPoint_mini_full_fs_max_min_combine"
mkdir $work_dir
cp -r * $work_dir/.
cd $work_dir

echo "Present working dir is  "$(pwd)

# Make sure the current working directory is in the python path.
export PYTHONPATH=$PYTHONPATH:$(pwd)
#export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
echo -e "Started  Script \n"

python -m torch.distributed.launch --nproc_per_node=4 --master_port=29503 ./tools/train.py \
            ./configs/waymo/pp/two_stage/waymo_centerpoint_pp_two_pfn_stride1_two_stage_bev_6epoch.py

echo -e "\n\n\n Finished Training Script \n"


echo -e "Started  Testing Script \n"

python -m torch.distributed.launch --nproc_per_node=4 --master_port=29503 ./tools/dist_test.py \
    /netscratch/jeevanandam/thesis/CenterPoint_results/work_dirs/mini/new/waymo_centerpoint_pp_two_pfn_stride1_two_stage_bev_6epoch_full_fs_max_min_combine/waymo_centerpoint_pp_two_pfn_stride1_two_stage_bev_6epoch.py \
    --work_dir /netscratch/jeevanandam/thesis/CenterPoint_results/work_dirs/mini/new/waymo_centerpoint_pp_two_pfn_stride1_two_stage_bev_6epoch_full_fs_max_min_combine \
    --checkpoint /netscratch/jeevanandam/thesis/CenterPoint_results/work_dirs/mini/new/waymo_centerpoint_pp_two_pfn_stride1_two_stage_bev_6epoch_full_fs_max_min_combine/latest.pth

echo -e "\n\n\n Finished Testing Script \n"
