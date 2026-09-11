import os
import numpy as np
import torch

from ultralytics import YOLO

import random
from datetime import datetime




device = 1


output_dir = "L3HarrisChadDroneDemo/model_results/synthetic_only2"


os.makedirs(output_dir, exist_ok=True)


train_data_yaml = "/data/od_datasets/l3harris_chad_drone/synthetic_data/flux1_lora2k_500_annotations/yolo/data.yaml"

test_data_yaml = "/data/od_datasets/l3harris_chad_drone/FLIR_drone_YRIKKA_frames_cropped_annotations/yolo/data.yaml"
test_split = "train"

model_name = "yolo11n.pt"
model = YOLO(model_name)

epochs = 50
batch_size = 16

project_dir = os.path.join(output_dir, "training_runs")
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

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
