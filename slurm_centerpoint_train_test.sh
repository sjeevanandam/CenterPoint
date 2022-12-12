#!/bin/bash

#to run parallel experiments make a copy everytime
# work_dir="/workspace/CenterPoint_mini_gnn_2l_kenc_64_d_384_full_fs"
# mkdir $work_dir
# cp -r * $work_dir/.
# cd $work_dir

echo "Present working dir is  "$(pwd)

# Make sure the current working directory is in the python path.
export PYTHONPATH=$PYTHONPATH:$(pwd)
#export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
echo -e "Started  Script \n"

python -m torch.distributed.launch --nproc_per_node=8 --master_port=29501 ./tools/train.py \
            ./configs/waymo/pp/two_stage/waymo_centerpoint_pp_two_pfn_stride1_two_stage_bev_6epoch_gnn.py

echo -e "\n\n\n Finished Training Script AND TRAIN ONLY\n"

work_dir="/netscratch/jeevanandam/thesis/CenterPoint_results/work_dirs/mini/new/waymo_centerpoint_pp_two_pfn_stride1_two_stage_bev_6epoch_gnn_fh_1920_to_256_6l_kenc_only_d_64_test"

echo -e "Started  Testing Script \n"

python -m torch.distributed.launch --nproc_per_node=8 --master_port=29501 ./tools/dist_test.py \
    $work_dir/waymo_centerpoint_pp_two_pfn_stride1_two_stage_bev_6epoch_gnn.py \
    --work_dir $work_dir \
    --checkpoint $work_dir/latest.pth

echo -e "\n\n\n Finished Testing Script \n"
