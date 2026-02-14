#!/bin/bash

# Initialize variables
DATASET=""
START_FOLD=""
END_FOLD=""
TASKS=""

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --start_fold) START_FOLD="$2"; shift ;;
        --end_fold) END_FOLD="$2"; shift ;;
        --tasks) TASKS="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Validate required arguments
if [[ -z "$DATASET" || -z "$START_FOLD" || -z "$END_FOLD" ]]; then
    echo "Usage: $0 --dataset <DATASET> --start_fold <START_FOLD_IDX> --end_fold <END_FOLD_IDX> [--tasks <task1,task2,...>]"
    exit 1
fi

# Activate Conda environment and set Python path
# Ensure conda is initialized for non-interactive shells
eval "$(conda shell.bash hook)"
conda activate RAG-Fuse
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the dataset-specific script
SCRIPT_PATH="run/${DATASET}.sh"

# Check if the script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script for dataset '$DATASET' not found at '$SCRIPT_PATH'"
    exit 1
fi

echo "Running $SCRIPT_PATH from fold $START_FOLD to fold $END_FOLD with tasks: ${TASKS:-ALL}..."

# Pass the arguments (including tasks) to the dataset script
bash "$SCRIPT_PATH" "$START_FOLD" "$END_FOLD" "$TASKS"