#!/usr/bin/env python
# coding=utf-8


import argparse
import logging
import math
import os
import random
from pathlib import Path
import random
import re
import tqdm
import datasets
import numpy as np
import nltk
from datasets import load_dataset, load_metric, load_from_disk


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="cnn_dailymail",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="3.0.0",
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_column",
        type=str,
        default="article",
    )

    parser.add_argument("--max-summary-length", type=int, default=148)
    return parser.parse_args()


def main():
    args = parse_args()

    raw_dataset = load_dataset(
        args.dataset_name, args.dataset_config_name, split="train"
    )
    inputs = raw_dataset[args.dataset_column]

    tokenizer = nltk.data.load(
        "tokenizers/punkt/english.pickle"
    )

    with open(f"./data/mono_{args.dataset_name}_{args.dataset_column}.txt", "w") as f:
        for text in tqdm(inputs):
            if args.dataset_name == "cnn_dailymail":
                text = text.strip("(CNN)  -- ")
                text = re.sub(".*\(CNN\)\s*--\s*", "", text)

            sents = tokenizer.tokenize(text)

            selected = []
            total_length = 0
            for sent in sents:
                length = len(nltk.word_tokenize(sent))  # Should have used BART tokenizer here! Inconsistent!
                if total_length + length <= args.max_summary_length:
                    f.write(sent + " ")
                    total_length += length
                else:
                    f.write("\n")
                    break


if __name__ == "__main__":
    main()
