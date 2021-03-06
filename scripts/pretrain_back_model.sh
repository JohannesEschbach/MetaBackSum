#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"
cd ../
accelerate test
accelerate launch backsum.py \
	  --model_name_or_path facebook/bart-base \
	  --dataset_name cnn_dailymail \
	  --dataset_config "3.0.0" \
	  --output_dir backmodels/backward_pretrained \
	  --num_warmup_steps 400 \
	  --max_source_length 148 \
	  --num_train_epochs 5 \
	  --max_target_length 1024 \
	  --per_device_batch_size 4 \
	  --text_column highlights \
	  --summary_column article \
	  --train_mode base \
	  --processed_data_dir data/backward \
	  --dataset_perc 0.03 \
	  --learning_rate 5e-5

