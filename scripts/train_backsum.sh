#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"
cd ../
accelerate test
accelerate launch backsum.py \
	  --model_name_or_path facebook/bart-base \
	  --dataset_name cnn_dailymail \
	  --mono_dataset_path data/mono_cnn_dailymail_text.txt \
	  --dataset_config "3.0.0" \
	  --output_dir models/$1 \
	  --back_model_output_dir backmodels/$1
	  --num_warmup_steps 400 \
	  --max_target_length 148 \
	  --num_train_epochs 5 \
	  --max_source_length 1024 \
	  --per_device_batch_size 2 \
	  --text_column article \
	  --summary_column highlights \
	  --train_mode $1 \
	  --processed_data_dir data/forward \
	  --processed_mono_data_dir data/mono \
	  --dataset_perc 0.03 \
	  --learning_rate 5e-5 \
  	  --back_learning_rate 5e-6

