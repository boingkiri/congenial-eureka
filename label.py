import os

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

def parse_args():
    parser = argparse.ArgumentParser(description="Labeling script")
    parser.add_argument(
        "--pretrained_yolo_path", type=str, default=None, help="Path to the pretrained YOLO model"
    )
    parser.add_argument(
        "--unseen_file_list_filename", type=str, default=None, help="File list to be labeled"
    )
    parser.add_argument(
        "--output_filename", type=str, default=None, help="Output filename for the labeled data"
    )
    args = parser.parse_args()
    return args

# Custom option
# pretrained_yolo_path = "/mnt/home/jeongjun/layout_diffusion/yolov11/generative_active_learning/250118_ms_coco_stuff_real_data_4p/weights/best.pt"
# unseen_file_list_filename  = "/mnt/home/jeongjun/layout_diffusion/yolov11/ms_coco_training_total_set.txt"


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

def main():
    args = parse_args()
    pretrained_yolo_path = args.pretrained_yolo_path
    unseen_file_list_filename = args.unseen_file_list_filename

    print("Pretrained YOLO path: ", pretrained_yolo_path)
    print("Unseen file list filename: ", unseen_file_list_filename)
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
            img_path = result.path
            img_filename = os.path.basename(img_path)
            img_metadata = find_from_obj(img_filename, annotation_obj)
            if img_metadata is None:
                continue

            new_coco_json_obj["images"].append(img_metadata)
            
            new_coco_json_annotations_element_format = {}
            new_coco_json_annotations_element_format["image_id"] = img_metadata["id"]
            new_coco_json_annotations_element_format["bbox"] = []
            new_coco_json_annotations_element_format["category_id"] = None
            new_coco_json_annotations_element_format["id"] = None

            for box in result.boxes:
                new_coco_json_annotations_element = copy.deepcopy(new_coco_json_annotations_element_format)
                xywh_box = box.xywh
                xywh_box = xywh_box.tolist()[0]

                coco_class_idx = convert_yolo_idx_to_coco_idx(int(box.cls.item()), yolo_categories, new_coco_json_obj["categories"])
                new_coco_json_annotations_element["bbox"] = xywh_box
                new_coco_json_annotations_element["category_id"] = coco_class_idx
                new_coco_json_annotations_element["id"] = global_count
                new_coco_json_annotations_element["conf"] = box.conf.item() # NEW!
                global_count += 1

                new_coco_json_obj["annotations"].append(new_coco_json_annotations_element)
        start += batch
        end += batch
        end = min(end, len(unseen_file_list))


    # with open("new_annotations_with_confidence.json", "w") as f:
    #     json.dump(new_coco_json_obj, f)
    # with open("new_annotations_with_cycle_0.json", "w") as f:
    #     json.dump(new_coco_json_obj, f)
    with open("pseudo_" + args.output_filename + ".json", "w") as f:
        json.dump(new_coco_json_obj, f)

if __name__ == "__main__":
    main()