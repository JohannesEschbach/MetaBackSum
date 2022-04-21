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
import datasets
import numpy as np
import nltk
from datasets import load_dataset, load_metric, load_from_disk


def parse_args():
    parser = argparse.ArgumentParser()
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
        "--dataset_column",
        type=str,
        default="highlights",
        choices=["article", "highlights"],
    )
    parser.add_argument(
        "--sample_mode",
        type=str,
        default="lead",
        choices=["lead", "random"],
    )

    parser.add_argument(
        "--max-summary-length",
        type=int,
        default=148,
    )
    return parser.parse_args()

def main():
    args = parse_args()
            
    raw_dataset = load_dataset(args.dataset_name, args.dataset_config_name, split="train")
    inputs = raw_dataset[args.dataset_column]

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') #Should have used BART tokenizer here! Inconsistent!

    with open(f"mono_{args.dataset_name}_{args.dataset_column}.txt", "w") as f:
        counter = 0
        for text in inputs[:1000]:
            if args.dataset_name == "cnn_dailymail":
                text = text.strip("(CNN)  -- ")
                text = re.sub(".*\(CNN\)\s*--\s*", "", text)

            sents = list(enumerate(tokenizer.tokenize(text)))                                                                                                                                          

            selected = []
            total_length = 0
            for sent in sents:
                length = len(nltk.word_tokenize(sent[1]))
                if total_length + length < args.max_summary_length:
                    selected.append(sent)
                    total_length += length
                else:
                    break


            selected.sort(key=lambda sent: sent[0])

            for sent in selected: f.write(sent[1] + " ")
            f.write("\n")
            print(counter, " / ", len(inputs))
            counter += 1


if __name__ == "__main__":
    main()

