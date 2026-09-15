import json
from pathlib import Path

from PIL import Image
from ultralytics.data.converter import convert_coco


dataset_root = Path(
    "data/f5ddba02-4295-48d9-8ad5-ba8583871323/"
    "bc6ba36c-aa02-4630-a0ea-4150b9b69e36/"
)

input_json = Path(
    "data/f5ddba02-4295-48d9-8ad5-ba8583871323/"
    "bc6ba36c-aa02-4630-a0ea-4150b9b69e36/"
    "coco_processed.json"
)

# Keep the repaired JSON in a directory containing only this COCO file.
labels_dir = dataset_root / "labels"
labels_dir.mkdir(parents=True, exist_ok=True)

repaired_json = labels_dir / "coco_processed.json"
image_dir = dataset_root / "Synthetic"
save_dir = dataset_root / "yolo_converted"

with input_json.open("r", encoding="utf-8") as file:
    coco_dict = json.load(file)

for image_record in coco_dict["images"]:
    image_path = image_dir / image_record["file_name"]

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image:
        image_record["width"], image_record["height"] = image.size

with repaired_json.open("w", encoding="utf-8") as file:
    json.dump(coco_dict, file, indent=4)
    file.write("\n")

convert_coco(
    labels_dir=str(labels_dir),
    save_dir=str(save_dir),
    cls91to80=False,
)

print(f"Repaired COCO JSON: {repaired_json}")
print(f"YOLO labels written to: {save_dir}")