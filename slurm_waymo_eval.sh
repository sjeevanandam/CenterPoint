#!/bin/bash
  
# Make sure the current working directory is in the python path.
export PYTHONPATH=$PYTHONPATH:$(pwd)
#export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
echo -e "Started Evaluation \n"

~/waymo-od/bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main \
    /netscratch/jeevanandam/thesis/CenterPoint_results/work_dirs/waymo_centerpoint_pp_two_pfn_stride1_two_stage_bev_6epoch/detection_pred.bin \
    /netscratch/jeevanandam/waymo_open_dataset_v_1_2_0_converted/gt_preds.bin \
    > /netscratch/jeevanandam/thesis/CenterPoint_results/work_dirs/waymo_centerpoint_pp_two_pfn_stride1_two_stage_bev_6epoch/results/results.txt

echo -e "\n\n\n Finished Script \n"
