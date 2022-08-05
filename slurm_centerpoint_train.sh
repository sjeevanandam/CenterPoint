#!/bin/bash

# Make sure the current working directory is in the python path.
export PYTHONPATH=$PYTHONPATH:$(pwd)
#export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
echo -e "Started  Script \n"

python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py \
            ./configs/waymo/pp/two_stage/waymo_centerpoint_pp_two_pfn_stride1_two_stage_bev_6epoch_fh.py

echo -e "\n\n\n Finished Trainig Script \n"

