import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from ultralytics import YOLO
import numpy as np
import time
import argparse


import json
import yaml

from mmengine import Config

from geodiffusion.geo_utils.data.new_coco_stuff import NewCOCOStuffDataset


from multiprocessing import Pool, Manager

import copy

# Custom option
pretrained_yolo_path = "/mnt/home/jeongjun/layout_diffusion/yolov11/generative_active_learning/250118_ms_coco_stuff_real_data_4p/weights/best.pt"
# unseen_file_list_filename = "/mnt/home/jeongjun/layout_diffusion/yolov11/unseen_log_cycle_1_004.yaml"
unseen_file_list_filename = "/mnt/home/jeongjun/layout_diffusion/datasets/250121_ms_coco_class_condition_disc/cycle_1/0/labels.txt"
dest_dir = "/mnt/home/jeongjun/layout_diffusion/yolov11/pseudo_label_yolo_format"


def find_from_obj(img_filename, shared_obj):
    # 1. Choose the image_metadata that is matched to the img_filename
    img_metadata = None
    for metadata in shared_obj["images"]:
        if metadata["file_name"] == img_filename:
            img_metadata = metadata
            break
    return img_metadata
    
def convert_yolo_idx_to_coco_idx(yolo_idx, yolo_categories, coco_categories):
    class_name = yolo_categories[yolo_idx]
    for coco_category in coco_categories:
        if coco_category["name"] == class_name:
            return coco_category["id"]
    return None

pretrained_yolo = YOLO(model=pretrained_yolo_path)

dataset_args = dict(
    prompt_version="v1", 
    num_bucket_per_side=[256, 256],
    foreground_loss_mode=True, 
    foreground_loss_weight=1.0,
    foreground_loss_norm=1.0,
    feat_size=64,
)
dataset_args_train = dict(
    uncond_prob=0.0,
)


pretrained_model_name = "KaiChen1998/geodiffusion-coco-stuff-512x512"

dataset_cfg = Config.fromfile("/mnt/home/jeongjun/layout_diffusion/geodiffusion_augmented_disc/geodiffusion/configs/data/new_coco_stuff_512x512.py")
dataset_cfg.data.train.update(**dataset_args)
dataset_cfg.data.train.update(**dataset_args_train)
dataset_cfg.data.train.pipeline[3]["flip_ratio"] = 0.0
dataset_cfg.data.val.pipeline[3]["flip_ratio"] = 0.0


# dataset_cfg.data.train.ann_file = "/mnt/home/jeongjun/layout_diffusion/geodiffusion_augmented_disc/coco-stuff-instance/instances_stuff_train2017.json"
# dataset_cfg.data.val.ann_file = "/mnt/home/jeongjun/layout_diffusion/geodiffusion_augmented_disc/coco-stuff-instance/instances_stuff_val2017.json"

annotation_obj = None
with open("/mnt/home/datasets/manual/coco_2017/annotations/instances_train2017.json", "r") as f:
    annotation_obj = json.load(f)

new_coco_json_obj = {
    "images": [],
    "annotations": [],
    "categories": annotation_obj["categories"]
}


print("Starting to run pretrained_yolo")
source_dir = "/mnt/home/datasets/manual/coco_2017/images/train2017"
if ".yaml" in unseen_file_list_filename:
    with open(unseen_file_list_filename, "r") as f:
        unseen_file_list_key_val = yaml.load(f, Loader=yaml.FullLoader)
    unseen_file_list = [os.path.join(source_dir, unseen_file_name) for unseen_file_name in unseen_file_list_key_val.keys()]
elif ".txt" in unseen_file_list_filename:
    with open(unseen_file_list_filename, "r") as f:
        unseen_file_list = [os.path.join(source_dir, unseen_file_name) for unseen_file_name in f.readlines()]
        unseen_file_list = [file_name.strip() for file_name in unseen_file_list]
else:
    print("Invalid file format")
    exit(1)

print("Unseen file list: ", len(unseen_file_list))
print(unseen_file_list[0])

# Yolo category
yolo_categories = {}
with open("/mnt/home/jeongjun/layout_diffusion/yolov11/data_yaml/1125_ms_coco_cycle_0/only_initial_and_vaal.yaml", "r") as f:
    yolo_categories = yaml.load(f, Loader=yaml.FullLoader)["names"]

batch = 512
start = 0
end = batch
global_count = 0
while start < len(unseen_file_list):
    print(f"Processing {start} to {end}")
    results = pretrained_yolo(unseen_file_list[start:end], stream=True)
    for result in results:
        # import pdb; pdb.set_trace()
        img_path = result.path
        img_filename = os.path.basename(img_path)
        yolo_label_filename = img_filename.replace(".jpg", ".txt")

        original_xy = result.orig_shape

        with open(os.path.join(dest_dir, yolo_label_filename), "w") as f:
            if len(result.boxes) == 0:
                f.write("")
            else:
                for box in result.boxes:
                    xywh_box = box.xywh
                    xywh_box = xywh_box.tolist()[0]
                    # Convert xywh to cxcywh
                    xywh_box[0] += xywh_box[2] / 2
                    xywh_box[1] += xywh_box[3] / 2
                    # Normalize
                    xywh_box[0] /= original_xy[1]
                    xywh_box[1] /= original_xy[0]
                    xywh_box[2] /= original_xy[1]
                    xywh_box[3] /= original_xy[0]

                    assert xywh_box[0] >= 0.0 and xywh_box[0] <= 1.0
                    assert xywh_box[1] >= 0.0 and xywh_box[1] <= 1.0
                    assert xywh_box[2] >= 0.0 and xywh_box[2] <= 1.0
                    assert xywh_box[3] >= 0.0 and xywh_box[3] <= 1.0

                    f.write(f"{int(box.cls.item())} {box.conf.item()} {' '.join([str(coord) for coord in xywh_box])}\n")

    start += batch
    end += batch
    end = min(end, len(unseen_file_list))


with open("new_annotations_with_confidence.json", "w") as f:
    json.dump(new_coco_json_obj, f)