# MetaBackSum

## Setup
0. Create a virtual environment and activate it.
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
For all models other than the baseline a pretrained backward model is required. Run [pretrain_back_model.sh](scripts/pretrain_back_model.sh).

## Train Models
The [scripts directory](scripts) contains all shell scripts for training.
To train the various configurations execute ```train_backsum.sh {MODE}```. 
Insert for ```{MODE}``` the name of the desired configuration: ```base```, ```nometa```, ```metahard``` or ```metadist```.

## Evaluate Models
Run [evaluate.sh](scripts/evaluate.sh) with the model path as argument to obtain Rouge score confidence intervals:
```
./scripts/evaluate.sh models/nometa
```
