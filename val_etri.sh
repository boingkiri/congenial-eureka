#!/bin/bash
project=generative_active_learning_etri
# name=250122_ms_coco_wo_acq_random_real_guided_coreset_cycle_1
# name=250124_etri_unguided_wo_acq_cycle_3
# name=250125_etri_coreset_cos_sim_w_acq_cycle_1
# name=250129_etri_coreset_cos_sim_wo_acq_cycle_1_again_2
# name=250129_etri_coreset_cos_sim_wo_acq_cycle_2
name=250129_etri_coreset_cos_sim_wo_acq_cycle_1_again
# device=0,1,2,3
device=3
# device=0
model=/mnt/home/jeongjun/layout_diffusion/yolov11/models/yolo11.yaml
data=/mnt/home/jeongjun/layout_diffusion/yolov11/data_yaml_etri/250124_etri_unguide/w_acq/cycle_1/unguide_1.yaml

for seed in 0 1 2; do
# for seed in 2; do
    name_seed=${name}_${seed}
    model=$project/$name_seed/weights/best.pt
    python val.py --project $project --name $name_seed --model $model --device $device --data $data
done

# device=0
# name=250127_etri_stuff_unguided_lr_0.005
# model=$project/$name/weights/best.pt
# python val.py --project $project --name $name --model $model --device $device --data $data 