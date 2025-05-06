from ultralytics import YOLO

import numpy as np
import os
import time
import argparse

os.environ["MKL_THREADING_LAYER"]="GNU"

# validation_skip = 200
validation_skip = 50

def on_train_epoch_end(trainer):
    """
    Callback that disables validation before a certain epoch.
    """
    skip_val_until_epoch = validation_skip  # Change this to your desired threshold
    if trainer.epoch + 1 < skip_val_until_epoch:
        print(f"[Callback] Skipping validation at epoch {trainer.epoch + 1}")
        trainer.args.val = False
        # trainer.validator.metrics = {}  # Reset any metrics if needed
        # trainer.console.info(f"[Callback] Skipping validation at epoch {trainer.epoch + 1}")
    else:
        trainer.args.val = True  # Re-enable validation


def on_train_start(trainer):
    print(f"On train START")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="model.pt path")
    parser.add_argument("--data", type=str, default="data/coco128.yaml", help="data.yaml path")
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="image size")
    parser.add_argument("--device", type=str, default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--project", type=str, default="runs/train", help="save to project/name")
    parser.add_argument("--name", type=str, default="exp", help="save to project/name")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--batch", type=int, default=512, help="batch size")
    return parser.parse_args()


def main():
    args = parse_args()

    print(args)

    model = YOLO(model=args.model)

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    args.device = args.device.split(",") if "," in args.device else args.device
    if isinstance(args.device, list):
        args.device = [int(x) for x in args.device]

    start_time = time.time()
    train_result = model.train(data=args.data, 
                            epochs=args.epochs, 
                            batch=args.batch, 
                            imgsz=args.imgsz, 
                            device=args.device, 
                            project=args.project, 
                            name=args.name,
                            lr0=args.lr)
    print(train_result)
    end_time = time.time()

    print("Time: ", end_time - start_time)


if __name__ == "__main__":
    main()