import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from ultralytics import YOLO
import numpy as np
import time
import argparse

import json
import yaml

from mmengine import Config

from geodiffusion.geo_utils.data.new_coco_stuff import NewCOCOStuffDataset

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision 

from multiprocessing import Pool, Manager

import copy
from tqdm import tqdm

from ultralytics.engine.predictor import BasePredictor
from ultralytics.utils import ops
from ultralytics.engine.results import Results
from ultralytics.models.yolo import YOLO, YOLOWorld
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel

# Custom option
pretrained_yolo_path = "/mnt/home/jeongjun/layout_diffusion/yolov11/disc_guidance_to_unlabeled/cycle_0/weights/best.pt"
conf_threshold = 0.25
pretrained_model_name = "KaiChen1998/geodiffusion-coco-stuff-512x512"

class CustomResults():
    def __init__(self, img, path, boxes):
        self.img = img
        self.path = path
        self.boxes = boxes
        self.speed = None
    def verbose(self):
        return ""

class CustomDetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model="yolo11n.pt", source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        # preds = ops.non_max_suppression(
        #     preds,
        #     self.args.conf,
        #     self.args.iou,
        #     agnostic=self.args.agnostic_nms,
        #     max_det=self.args.max_det,
        #     classes=self.args.classes,
        # )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        # preds = [pred[0] for pred in preds]
        preds = preds[0]
        print(len(preds))
        print(len(orig_imgs))
        print(len(self.batch[0]))
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            # results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
            # results.append((orig_img, img_path, pred))
            results.append(CustomResults(orig_img, img_path, pred))
        return results


class CustomYOLO(YOLO):
    """YOLO (You Only Look Once) object detection model."""

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": CustomDetectionPredictor,
            },
        }

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

pretrained_yolo = CustomYOLO(model=pretrained_yolo_path)


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




annotation_obj = None
with open("/mnt/home/datasets/manual/coco_2017/annotations/instances_train2017.json", "r") as f:
    annotation_obj = json.load(f)


print("Starting to run pretrained_yolo")
source_dir = "/mnt/home/datasets/manual/coco_2017/images/train2017"
image_metadata_list = annotation_obj["images"]
file_list = [os.path.join(source_dir, image_metadata["file_name"]) for image_metadata in image_metadata_list]
file_list = sorted(file_list)

print("Unseen file list: ", len(file_list))
print(file_list[0])

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((512, 512)),
    # torchvision.transforms.Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
])
# transform = torchvision.transforms.Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])


class CustomImageDataset(Dataset):
    def __init__(self, img_file_path_list, transform=None, target_transform=None):
        self.img_file_path_list = img_file_path_list
        self.transform = transform

    def __len__(self):
        return len(self.img_file_path_list)

    def __getitem__(self, idx):
        image_file_path = self.img_file_path_list[idx]
        return image_file_path

def collate_fn(batch):
    return {"path": batch}

image_dataset = CustomImageDataset(file_list, transform=transform)
# dataloader = DataLoader(image_dataset, batch_size=512, shuffle=False, num_workers=8)
dataloader = DataLoader(image_dataset, batch_size=512, shuffle=False, num_workers=8, collate_fn=collate_fn)


# Yolo category
yolo_categories = {}

with open("/mnt/home/jeongjun/layout_diffusion/yolov11/data_yaml/1125_ms_coco_cycle_0/only_initial_and_vaal.yaml", "r") as f:
    yolo_categories = yaml.load(f, Loader=yaml.FullLoader)["names"]

start_time = time.time()

max_entropy_result = {}
for data in dataloader:
    path = data["path"]
    results = pretrained_yolo.predict(path)
    for result in results:
        boxes = result.boxes
        boxes_confs = boxes[4:]
        ## threshold
        boxes_thr = torch.where(boxes_confs > conf_threshold, 1, 0)
        selected_box_index = torch.where(boxes_thr == 1)
        selected_boxes = boxes_confs[:, selected_box_index[1]]

        # Calculate entropy
        selected_boxes = selected_boxes / torch.sum(selected_boxes, dim=0)
        entropy = -torch.sum(selected_boxes * torch.log(selected_boxes), dim=0)
        if entropy.size(0) == 0:
            continue
        max_entropy = torch.max(entropy)
        max_entropy_result[result.path.split("/")[-1]] = max_entropy.cpu().item()

end_time = time.time()
print("Elapsed time: ", end_time - start_time)
with open("max_entropy_result.json", "w") as f:

    json.dump(max_entropy_result, f)