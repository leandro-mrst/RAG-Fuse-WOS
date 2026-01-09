#!/bin/bash

# Load .env if it exists
if [ -f .env ]; then
  set -a          # automatically export all variables
  source .env
  set +a
else
  echo ".env file not found"
  exit 1
fi

# Check if the correct number of arguments is provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <DATASET> <START_FOLD_IDX> <END_FOLD_IDX>"
    exit 1
fi

# Assign variables
DATASET=$1
START_FOLD=$2
END_FOLD=$3

# Activate Conda environment and set Python path
# Ensure conda is initialized for non-interactive shells
eval "$(conda shell.bash hook)"
conda activate temp
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the dataset-specific script for each fold index in the given range
SCRIPT_PATH="run/${DATASET}.sh"

# Check if the script exists before running it
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script for dataset '$DATASET' not found at '$SCRIPT_PATH'"
    exit 1
fi

# run the script
echo "Running $SCRIPT_PATH from fold $START_FOLD to fold $END_FOLD..."
bash "$SCRIPT_PATH" "$START_FOLD" "$END_FOLD"









