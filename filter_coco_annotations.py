#!/usr/bin/env python3
"""Keep only category_id == 1 annotations from a COCO JSON file."""

import argparse
import json
from pathlib import Path


def filter_coco_annotations(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as input_file:
        coco_dict = json.load(input_file)

    if not isinstance(coco_dict, dict):
        raise ValueError("The COCO JSON root must be a dictionary.")
    if not isinstance(coco_dict.get("annotations"), list):
        raise ValueError("The COCO JSON must contain an 'annotations' list.")

    coco_dict["annotations"] = [
        annotation
        for annotation in coco_dict["annotations"]
        if annotation.get("category_id") == 1
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(coco_dict, output_file, indent=4)
        output_file.write("\n")

    print(f"Wrote {len(coco_dict['annotations'])} annotations to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter a COCO JSON file to category_id == 1 annotations."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=Path(
            "data/f5ddba02-4295-48d9-8ad5-ba8583871323/"
            "08788e35-b887-4d04-9a7f-40b900d15569/coco.json"
        ),
        help="Path to the input coco.json file.",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        type=Path,
        default=Path(
            "data/f5ddba02-4295-48d9-8ad5-ba8583871323/"
            "08788e35-b887-4d04-9a7f-40b900d15569/coco_processed.json"
        ),
        help="Path for the filtered COCO JSON output.",
    )
    args = parser.parse_args()
    filter_coco_annotations(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
