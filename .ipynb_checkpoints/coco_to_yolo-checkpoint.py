from pathlib import Path


'''
# Demo.

# Multiple COCO boxes for one image
boxes = [
    (0, [100, 50, 200, 120]),
    (2, [350, 180, 90, 160]),
    (1, [20, 300, 75, 80]),
]

coco_boxes_to_yolo_txt(
    boxes=boxes,
    image_width=640,
    image_height=480,
    output_path="labels/image_001.txt",
)
'''

def coco_boxes_to_yolo_txt(
    boxes,
    image_width,
    image_height,
    output_path,
):
    """
    Convert multiple COCO boxes to an Ultralytics YOLO label file.

    Parameters
    ----------
    boxes : list[tuple[int, list[float]]]
        Each element is: (class_id, [x_min, y_min, width, height]).
    image_width, image_height : int
        Image dimensions in pixels.
    output_path : str | Path
        Destination .txt path.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    label_lines = []

    for class_id, bbox in boxes:
        x_min, y_min, width, height = bbox

        x_center = (x_min + width / 2) / image_width
        y_center = (y_min + height / 2) / image_height
        norm_width = width / image_width
        norm_height = height / image_height

        # Ultralytics format:
        # class_id x_center y_center width height
        label_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} "
            f"{norm_width:.6f} {norm_height:.6f}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""))
