#!/usr/bin/env python3

import argparse
from pathlib import Path

import boto3

'''
# To run.
python -m pip install boto3
python download_s3_bucket.py my-bucket ./my-bucket-download

python download_s3_bucket.py my-bucket ./download \
  --profile production \
  --prefix datasets/2026/
'''


def safe_destination(root: Path, key: str) -> Path:
    """Map an S3 key into root without allowing path traversal."""
    destination = (root / key).resolve()

    try:
        destination.relative_to(root)
    except ValueError:
        raise ValueError(f"Unsafe S3 object key: {key!r}")

    return destination


def download_bucket(
    bucket: str,
    output_dir: Path,
    prefix: str = "",
    profile: str | None = None,
) -> None:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    session = boto3.Session(profile_name=profile)
    s3 = session.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    downloaded = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            # S3 console-style directory marker
            if key.endswith("/"):
                safe_destination(output_dir, key).mkdir(
                    parents=True, exist_ok=True
                )
                continue

            destination = safe_destination(output_dir, key)
            destination.parent.mkdir(parents=True, exist_ok=True)

            print(f"s3://{bucket}/{key} -> {destination}")
            s3.download_file(bucket, key, str(destination))
            downloaded += 1

    print(f"Downloaded {downloaded} objects to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download every current object from an S3 bucket."
    )
    parser.add_argument("bucket", help="S3 bucket name")
    parser.add_argument("output", type=Path, help="Local destination directory")
    parser.add_argument(
        "--prefix",
        default="",
        help="Download only keys beginning with this prefix",
    )
    parser.add_argument(
        "--profile",
        help="Optional AWS CLI profile name",
    )
    args = parser.parse_args()

    download_bucket(
        bucket=args.bucket,
        output_dir=args.output,
        prefix=args.prefix,
        profile=args.profile,
    )


if __name__ == "__main__":
    main()