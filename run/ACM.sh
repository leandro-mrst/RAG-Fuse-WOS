#!/bin/bash

START_FOLD=$1
END_FOLD=$2
TASKS_ARG=$3

# Helper function to check if a specific task is in the requested tasks list
# Usage: if should_run "task_name"; then ... fi
should_run() {
    local task_name=$1
    # Check if TASKS_ARG contains the task_name (surrounded by commas to avoid partial matches)
    if [[ ",$TASKS_ARG," == *",$task_name,"* ]]; then
        return 0
    else
        return 1
    fi
}

# overrides
data=ACM
model=SparseRetrieverBERT

text_max_length=256
label_max_length=256
label_enhancement=LLM
text_features_source=TXT
experiment=ZS

## sparse_retrieve
if should_run "sparse_retrieve"; then
  for fold_idx in $(seq $START_FOLD $END_FOLD);
  do
    time_start=$(date '+%Y-%m-%d %H:%M:%S')
    python main.py \
      tasks=[sparse_retrieve] \
      model=BM25 \
      data=$data \
      data.text_features_source=$text_features_source \
      data.folds=[$fold_idx]
    time_end=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$time_start,$time_end" > resource/time/sparse_retrieve_${data}_${fold_idx}.tmr
  done
fi

## elsa
if should_run "elsa"; then
  for fold_idx in $(seq $START_FOLD $END_FOLD);
  do
    time_start=$(date '+%Y-%m-%d %H:%M:%S')
    python main.py \
      tasks=[elsa] \
      model=$model \
      model.name=${experiment}_${model} \
      data=$data \
      data.folds=[$fold_idx]
    time_end=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$time_start,$time_end" > resource/time/sparse_retrieve_${data}_${fold_idx}.tmr
  done
fi

## prompt_opt
if should_run "prompt_opt"; then
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py \
    tasks=[prompt_opt] \
    data=$data \
    data.text_features_source=$text_features_source
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/prompt_opt_${data}_${fold_idx}.tmr
fi

## label_desc
if should_run "label_desc"; then
  for fold_idx in $(seq $START_FOLD $END_FOLD);
  do
    time_start=$(date '+%Y-%m-%d %H:%M:%S')
    python main.py \
      tasks=[label_desc] \
      data=$data \
      data.text_features_source=$text_features_source \
      data.folds=[$fold_idx]
    time_end=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$time_start,$time_end" > resource/time/label_desc_${data}_${fold_idx}.tmr
  done
fi

## overlap_labels
if should_run "overlap_labels"; then
  for fold_idx in $(seq $START_FOLD $END_FOLD);
  do
    time_start=$(date '+%Y-%m-%d %H:%M:%S')
    python main.py \
      tasks=[overlap_labels] \
      trainer.max_epochs=5 \
      trainer.patience=3 \
      model=$model \
      model.name=${label_enhancement}_${model} \
      data=$data \
      data.text_max_length=$text_max_length \
      data.label_max_length=$label_max_length \
      data.label_enhancement=$label_enhancement \
      data.text_features_source=$text_features_source \
      data.batch_size=64 \
      data.num_workers=12 \
      data.folds=[$fold_idx]
    time_end=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$time_start,$time_end" > resource/time/overlap_labels_${label_enhancement}_${model}_${data}_${fold_idx}.tmr
  done
fi

# dense_retrieve fit
if should_run "fit"; then
  for fold_idx in $(seq $START_FOLD $END_FOLD);
  do
    time_start=$(date '+%Y-%m-%d %H:%M:%S')
    python main.py \
      tasks=[fit] \
      trainer.max_epochs=5 \
      trainer.patience=3 \
      trainer.min_delta=0.001 \
      trainer.monitor="train_LOSS" \
      trainer.mode="min" \
      model=$model \
      model.name=${experiment}_${model} \
      data=$data \
      data.text_max_length=$text_max_length \
      data.label_max_length=$label_max_length \
      data.label_enhancement=$label_enhancement \
      data.text_features_source=$text_features_source \
      data.batch_size=32 \
      data.num_workers=12 \
      data.folds=[$fold_idx]
    time_end=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$time_start,$time_end" > resource/time/fit_LLM_${model}_${data}_${fold_idx}.tmr
  done
fi

# dense_retrieve predict
if should_run "predict"; then
  for fold_idx in $(seq $START_FOLD $END_FOLD);
  do
    time_start=$(date '+%Y-%m-%d %H:%M:%S')
    python main.py \
      tasks=[predict] \
      trainer.max_epochs=5 \
      trainer.patience=3 \
      model=$model \
      model.name=${experiment}_${model} \
      data=$data \
      data.text_max_length=$text_max_length \
      data.label_max_length=$label_max_length \
      data.label_enhancement=$label_enhancement \
      data.text_features_source=$text_features_source \
      data.batch_size=64 \
      data.num_workers=12 \
      data.folds=[$fold_idx]
    time_end=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$time_start,$time_end" > resource/time/predict_LLM_${model}_${data}_${fold_idx}.tmr
  done
fi

# oracle dime
if should_run "oracle_dime"; then
  for fold_idx in $(seq $START_FOLD $END_FOLD);
  do
    python main.py \
      tasks=[oracle_dime] \
      model=$model \
      model.name=${experiment}_${model} \
      data=$data \
      data.folds=[$fold_idx]
  done
fi

## proxy dime
if should_run "proxy_dime"; then
  for fold_idx in $(seq $START_FOLD $END_FOLD);
  do
    python main.py \
      tasks=[proxy_dime] \
      dime.proxy.num_neighbors=2 \
      model=$model \
      model.name=${experiment}_${model} \
      data=$data \
      data.folds=[$fold_idx]
  done
fi

## dense_retrieve eval
if should_run "eval"; then
  for fold_idx in $(seq $START_FOLD $END_FOLD);
  do
    time_start=$(date '+%Y-%m-%d %H:%M:%S')
    python main.py \
      tasks=[eval] \
      trainer.max_epochs=5 \
      trainer.patience=3 \
      model=$model \
      model.name=${experiment}_${model} \
      data=$data \
      data.text_max_length=$text_max_length \
      data.label_max_length=$label_max_length \
      data.label_enhancement=$label_enhancement \
      data.text_features_source=$text_features_source \
      data.batch_size=64 \
      data.num_workers=12 \
      data.folds=[$fold_idx]
    time_end=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$time_start,$time_end" > resource/time/eval_LLM_${model}_${data}_${fold_idx}.tmr
  done
fi

## fuse
if should_run "fuse"; then
  for fold_idx in $(seq $START_FOLD $END_FOLD);
  do
    time_start=$(date '+%Y-%m-%d %H:%M:%S')
    python main.py \
      tasks=[fuse] \
      model=$model \
      model.name=LLM_RetrieverBERT \
      data=$data \
      data.text_features_source=$text_features_source \
      data.folds=[$fold_idx]
    time_end=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$time_start,$time_end" > resource/time/fuse_LLM_${model}_${data}_${fold_idx}.tmr
  done
fi

## aggregate
if should_run "aggregate"; then
  for fold_idx in $(seq $START_FOLD $END_FOLD);
  do
    time_start=$(date '+%Y-%m-%d %H:%M:%S')
    python main.py \
      tasks=[aggregate] \
      model=$model \
      model.name=LLM_RetrieverBERT \
      data=$data \
      data.text_max_length=$text_max_length \
      data.label_max_length=$label_max_length \
      data.label_enhancement=$label_enhancement \
      data.text_features_source=$text_features_source \
      data.batch_size=64 \
      data.num_workers=12 \
      data.folds=[$fold_idx]
    time_end=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$time_start,$time_end" > resource/time/aggregate_LLM_${model}_${data}_${fold_idx}.tmr
  done
fi