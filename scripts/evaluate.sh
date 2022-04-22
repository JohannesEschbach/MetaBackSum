#!/bin/bash

accelerate test
accelerate launch evaluate.py \
          --model_name_or_path $1 \
          --dataset_name cnn_dailymail \
          --dataset_config "3.0.0" \
          --max_source_length 1024 \
          --max_target_length 148 \
          --processed_data_dir ./data/forward \
          --per_device_batch_size 4 \
