#!/usr/bin/env python
# coding=utf-8
# Roughly based on the Accelerator Example script provided by Huggingface:
# https://github.com/huggingface/transformers/tree/master/examples/pytorch/summarization

import argparse
import logging
import os
import random
import statistics

import datasets
import nltk
import numpy as np
import torch
from datasets import load_dataset, load_metric, load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a summarization task"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="article",
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default="highlights",
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        type=bool,
        default=None,
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=148,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--processed_data_dir", type=str, required=True)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    accelerator = Accelerator()
    logger.info(accelerator.state)
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    config = AutoConfig.from_pretrained(args.model_name_or_path)

    # Load tokenizers
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=True
    )

    # Load forward model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    model.resize_token_embeddings(len(tokenizer))

    # Preprocess the Summarization Dataset
    column_names = raw_datasets["test"].column_names


    def preprocess_function(examples):
        inputs = examples[args.text_column]
        targets = examples[args.summary_column]
        model_inputs = tokenizer(
            inputs,
            max_length=args.max_source_length,
            padding="max_length",
            truncation=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=args.max_target_length,
                padding="max_length",
                truncation=True,
            )
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if not os.path.exists(args.processed_data_dir) or args.overwrite_cache:
        with accelerator.main_process_first():
            os.makedirs(args.processed_data_dir, exist_ok=True)
            processed_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
            processed_datasets.save_to_disk(args.processed_data_dir)
    else:
        processed_datasets = load_from_disk(args.processed_data_dir)

    test_dataset = processed_datasets["test"]

    # Collators
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    random.seed(42)
    torch.manual_seed(42)
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=args.per_device_batch_size,
    )

    # Prepare everything with our `accelerator`.
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    # Metric
    metric = load_metric("rouge")

    # Test!
    total_batch_size = args.per_device_batch_size * accelerator.num_processes
    logger.info("***** Running testing *****")
    logger.info(f"  Num examples = {len(test_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(len(test_dataloader)), disable=not accelerator.is_local_main_process
    )

    model.eval()

    gen_kwargs = {"max_length": args.max_target_length}
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():

            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"], attention_mask=batch["attention_mask"], **gen_kwargs
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = batch["labels"]

            labels = accelerator.pad_across_processes(
                batch["labels"], dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            labels = accelerator.gather(labels).cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            decoded_preds = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(
                decoded_preds, decoded_labels
            )

            metric.add_batch(predictions=decoded_preds, references=decoded_labels)

            progress_bar.update(1)
    
    # Output [low, mid, high]
    result = metric.compute(use_stemmer=True)
    result = {key: [value.low.fmeasure * 100, value.mid.fmeasure * 100, value.high.fmeasure * 100] for key, value in result.items()}
    result = {k: [round(item, 4) for item in v] for k, v in result.items()}
    logger.info(result)


if __name__ == "__main__":
    main()
