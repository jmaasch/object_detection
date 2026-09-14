#!/usr/bin/env bash
set -euo pipefail

# Default values matching the existing script behavior.
MODEL_NAME="yolo11n.pt"
N_EPOCHS=50
BATCH_SIZE=16
OUTPUT_DIR="output"
TRAIN_YAML="train_data.yaml"
TEST_YAML="test_data.yaml"
TEST_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name|--model-name|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --n_epochs|--epochs)
            N_EPOCHS="$2"
            shift 2
            ;;
        --batch_size|--batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --output_dir|--output-directory)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --train_yaml|--train-yaml)
            TRAIN_YAML="$2"
            shift 2
            ;;
        --test_yaml|--test-yaml)
            TEST_YAML="$2"
            shift 2
            ;;
        --test_only|--test-only)
            TEST_ONLY=true
            shift
            ;;
        --test_only=*)
            TEST_ONLY="${1#*=}"
            shift
            ;;
        --help|-h)
            echo "Usage: ./run_yolo.sh [flags]"
            echo "  --model_name <model>"
            echo "  --n_epochs <int>"
            echo "  --batch_size <int>"
            echo "  --output_dir <path>"
            echo "  --train_yaml <path>"
            echo "  --test_yaml <path>"
            echo "  --test_only [true|false]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Run with --help for supported flags."
            exit 2
            ;;
    esac
done

if [[ "$TEST_ONLY" == "true" || "$TEST_ONLY" == "True" || "$TEST_ONLY" == "1" ]]; then
    TEST_ONLY_ARG="--test_only"
else
    TEST_ONLY_ARG=""
fi

python train_and_test_yolo.py \
  --model_name "$MODEL_NAME" \
  --n_epochs "$N_EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --output_dir "$OUTPUT_DIR" \
  --train_yaml "$TRAIN_YAML" \
  --test_yaml "$TEST_YAML" \
  ${TEST_ONLY_ARG}
