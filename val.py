from ultralytics import YOLO
import numpy as np
import os
import time
import argparse

os.environ["MKL_THREADING_LAYER"]="GNU"

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="model.pt path")
    parser.add_argument("--data", type=str, default="data/coco128.yaml", help="data.yaml path")
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--batch", type=int, default=512, help="batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="image size")
    parser.add_argument("--device", type=str, default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--project", type=str, default="runs/train", help="save to project/name")
    parser.add_argument("--name", type=str, default="exp", help="save to project/name")
    return parser.parse_args()


def main():
    args = parse_args()

    print(args)

    model = YOLO(model=args.model)

    args.device = args.device.split(",") if "," in args.device else args.device
    if isinstance(args.device, list):
        args.device = [int(x) for x in args.device]

    start_time = time.time()
    val_result = model.val(data=args.data, 
                            device=args.device,)
    # print(val_result)
    end_time = time.time()


if __name__ == "__main__":
    main()