# MetaBackSum

## Setup
1. Install the requirements with ```pip install -r requirements.txt```
2. Setup Huggingface's Accelerate by running ```accelerate config``` and answering the questions asked.

## Create Summary Corpus
To create the corpus of pseudo summaries run
```
python dataset_generator.py \
  --dataset_name cnn_dailymail \
  --dataset_config_name "3.0.0" \
  --dataset_column "article"
```
Feel free to use any other dataset listed on https://huggingface.co/datasets. 

## Pretrain Backward Model
Run 
