#!/bin/bash
# project=generative_active_learning

# name=250410_ms_coco_stuff_adv_weight_001_cycle_1
name=250505_ms_coco_stuff_multi_label_classifier_cycle_1_scale_20

device=4
model=/mnt/home/jeongjun/layout_diffusion/yolov11/models/yolo11.yaml
data=/mnt/home/jeongjun/layout_diffusion/yolov11/data_yaml/1205_ms_coco_stuff_guidance_only_bbox/only_bbox.yaml
project=generative_active_learning

for seed in 0 1 2; do
# for seed in 2; do
    name_seed=${name}_${seed}
    model=$project/$name_seed/weights/best.pt
    python val.py --project $project --name $name_seed --model $model --device $device --data $data
done


# name_seed=${name}
# model=$project/$name_seed/weights/best.pt
# python val.py --project $project --name $name_seed --model $model --device $device --data $data


# name=250130_baseline
# name_seed=${name}
# model=/mnt/home/jeongjun/layout_diffusion/paper/generative_active_learning/cycle0_03/weights/best.pt
# project=generative_active_learning
# device=3
# python val.py --project $project --name $name_seed --model $model --device $device --data $data