import argparse
import os
import numpy as np
import torch

from ultralytics import YOLO

import random
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Train and validate a YOLO model.")
    parser.add_argument("--model_name", default="yolo11n.pt", help="Model checkpoint file or YOLO model name.")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--output_dir", default="L3HarrisChadDroneDemo/model_results/synthetic_only2", help="Directory where artifacts will be written.")
    parser.add_argument("--train_yaml", default="flux1_lora_2k_500_yolo/data.yaml", help="Path to the training data YAML.")
    parser.add_argument("--test_yaml", default="FLIR_drone_YRIKKA_frames_cropped_annotations/yolo/data.yaml", help="Path to the validation/test data YAML.")
    parser.add_argument("--test_only", action="store_true", help="Skip training and validate an existing model on test data only.")
    return parser.parse_args()


args = parse_args()

#device = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

train_data_yaml = args.train_yaml
test_data_yaml = args.test_yaml
test_split = "train"

model_name = args.model_name
model = YOLO(model_name)

epochs = args.n_epochs
batch_size = args.batch_size

project_dir = os.path.join(output_dir, "training_runs")
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

if not args.test_only:
    model.train(
        data=train_data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        device=device,
        workers=8,
        project=project_dir,
        name=run_name,
        exist_ok=True,
        cache=False
    )

    model_save_path = os.path.join(output_dir, "model.pt")
    model.save(model_save_path)

    print(f"MODEL SAVED AT: {os.path.abspath(model_save_path)}")
    print(f"TRAINING OUTPUTS SAVED AT: {os.path.abspath(os.path.join(project_dir, run_name))}")

metrics = model.val(data=test_data_yaml, split=test_split, imgsz=640, device=device)
print(f"Validation AFTER mAP@0.5: {metrics.box.map50:.4f}")
print(f"Validation AFTER mAP@0.75: {metrics.box.map75:.4f}")
print(f"Validation AFTER mAP@50-0.95: {metrics.box.map:.4f}")

results_file = os.path.join(output_dir, "validation_results.txt")
with open(results_file, "w") as f:
    f.write(f"Validation Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 50 + "\n")
    f.write(f"Dataset: {test_data_yaml}\n")
    f.write(f"Model: {model_name}\n")
    f.write(f"Epochs: {epochs}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Image Size: 640\n")
    f.write("=" * 50 + "\n")
    f.write(f"{'Class':<20} {'AP@0.5':>10} {'AP@0.75':>10} {'AP@0.5-0.95':>12}\n")
    f.write("-" * 55 + "\n")
    f.write(f"{'all':<20} {metrics.box.map50:>10.4f} {metrics.box.map75:>10.4f} {metrics.box.map:>12.4f}\n")
    class_names = metrics.names
    for i, cls_idx in enumerate(metrics.box.ap_class_index):
        name = class_names[cls_idx]
        ap50 = metrics.box.ap50[i]
        ap75 = metrics.box.all_ap[i, 5]
        ap = metrics.box.ap[i]
        f.write(f"  {name:<18} {ap50:>10.4f} {ap75:>10.4f} {ap:>12.4f}\n")

print(f"Validation results saved to: {os.path.abspath(results_file)}")
