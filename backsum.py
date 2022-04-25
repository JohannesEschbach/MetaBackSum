#!/usr/bin/env python
# coding=utf-8
# Roughly based on the Accelerator Example script provided by Huggingface:
# https://github.com/huggingface/transformers/tree/master/examples/pytorch/summarization

import argparse
import logging
import math
import os
import random

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
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
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
        "--mono_dataset_path",
        type=str,
        default=None,
        help="The name of the backward dataset to use.",
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
        "--back_model_path",
        type=str,
        help="Path to pretrained backward model.",
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
        "--per_device_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--back_learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="polynomial",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--back_model_output_dir",
        type=str,
        default=None,
        help="Where to store the final backward model.",
    )
    parser.add_argument("--processed_data_dir", type=str)
    parser.add_argument("--processed_mono_data_dir", type=str)
    parser.add_argument("--dataset_perc", type=float, default=0.03)
    parser.add_argument(
        "--train_mode", type=str, choices=["base", "nometa", "metahard", "metadist"]
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    BACK = args.train_mode != "base"

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

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.back_model_output_dir is not None:
            os.makedirs(args.back_model_output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    if BACK:
        mono_dataset = load_dataset("text", data_files=args.mono_dataset_path)

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    if BACK:
        back_config = AutoConfig.from_pretrained(args.back_model_path)

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

    # Load backward model
    if BACK:
        back_model = AutoModelForSeq2SeqLM.from_pretrained(
            args.back_model_path,
            from_tf=bool(".ckpt" in args.back_model_path),
            config=back_config,
        )
        back_model.resize_token_embeddings(len(tokenizer))

    # Preprocess the Summarization Dataset
    column_names = raw_datasets["train"].column_names


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

    # Scarcify Dataset
    random.seed(42)
    train_dataset = processed_datasets["train"].select(
        random.sample(
            range(len(processed_datasets["train"])),
            int(args.dataset_perc * len(processed_datasets["train"])),
        )
    )

    # Reduce evaluation-set size to speed things up
    eval_dataset = processed_datasets["validation"].select(
        random.sample(
            range(len(processed_datasets["validation"])),
            int(0.25 * len(processed_datasets["validation"])),
        )
    )

    # Preprocess Summary Corpus
    if BACK:

        def mono_preprocess_function(inputs):
            return tokenizer(
                inputs["text"],
                max_length=args.max_target_length,
                padding="max_length",
                truncation=True,
            )

        if not os.path.exists(args.processed_mono_data_dir) or args.overwrite_cache:
            with accelerator.main_process_first():
                os.makedirs(args.processed_mono_data_dir, exist_ok=True)
                processed_mono_dataset = mono_dataset.map(
                    mono_preprocess_function,
                    batched=True,
                    remove_columns="text",
                    num_proc=args.preprocessing_num_workers,
                    load_from_cache_file=not args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )
                processed_mono_dataset.save_to_disk(args.processed_mono_data_dir)
        else:
            processed_mono_dataset = load_from_disk(args.processed_mono_data_dir)
        processed_mono_dataset = processed_mono_dataset["train"]

    # Collators
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    if BACK:
        mono_data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=back_model,
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

    torch.manual_seed(42)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=args.per_device_batch_size
    )

    if BACK:
        mono_dataloader = DataLoader(
            processed_mono_dataset,
            shuffle=True,
            collate_fn=mono_data_collator,
            batch_size=args.per_device_batch_size,
        )

    # Optimizers
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    if BACK:
        back_optimizer = AdamW(
            back_model.parameters(), lr=args.learning_rate * 0.1
        )  # 10 times lower lr

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    if BACK:
        mono_dataloader, back_model, back_optimizer = accelerator.prepare(
            mono_dataloader, back_model, back_optimizer
        )

    # Scheduler and math around the number of training steps.
    max_train_steps = args.num_train_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    if BACK:
        back_lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=back_optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=max_train_steps,
        )

    # Metric
    metric = load_metric("rouge")

    # Train!
    total_batch_size = args.per_device_batch_size * accelerator.num_processes
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )

    # Logger Variables
    completed_steps = 0
    loss_on_generated_for_logger = 0
    loss_for_logger = 0
    back_loss_for_logger = 0

    # Reinforcement Reward
    cos = torch.nn.CosineSimilarity(dim=-1)
    reward_mv_avg = 0.0

    # Maximum Entropy
    vocab_size = accelerator.unwrap_model(back_model).config.vocab_size
    logger.info(f" Vocab size: {vocab_size}")
    max_ent = math.log(1 / vocab_size)

    # Generation Keywords
    back_gen_kwargs = {"max_length": args.max_source_length}
    gen_kwargs = {"max_length": args.max_target_length}

    if BACK:
        mono_iterator = iter(mono_dataloader)
    for epoch in range(args.num_train_epochs):
        model.train()
        if BACK:
            back_model.train()
        for batch in train_dataloader:

            if BACK:
                # STEP 1: SAMPLE SYNTHETIC DOCUMENT FROM SUMMARY
                try:
                    mono_batch = next(mono_iterator)
                except StopIteration:
                    mono_iterator = iter(mono_dataloader)
                    mono_batch = next(mono_iterator)

                backward_logits = back_model(**mono_batch).logits

                softmax_probs = torch.nn.functional.gumbel_softmax(
                    backward_logits, hard=True, dim=-1
                )
                generated_tokens = softmax_probs.argmax(-1)

                generated_batch = {}
                generated_batch["labels"] = mono_batch["input_ids"].detach().clone()
                generated_batch["input_ids"] = generated_tokens.detach().clone()
                generated_batch["labels"] = torch.where(
                    generated_batch["labels"] != tokenizer.pad_token_id,
                    generated_batch["labels"],
                    -100,
                )

                # MAKE ATTENTION MASK
                generated_batch["attention_mask"] = torch.where(
                    generated_batch["input_ids"] != tokenizer.pad_token_id, 1, 0
                )

                # STEP 2: UPDATE FORWARD MODEL WITH ARTIFICIAL DATA PAIR
                optimizer.zero_grad()
                outputs = model(**generated_batch)
                loss = outputs.loss
                accelerator.backward(loss)
                # Save Gradient
                grad_vec_old = torch.cat(
                    tuple(
                        torch.squeeze(
                            torch.reshape(
                                par.grad.detach().clone(), (1, torch.numel(par))
                            )
                        )
                        for par in list(model.parameters())
                    )
                )
                loss_on_generated_for_logger += loss
                optimizer.step()
                lr_scheduler.step()

            # STEP 3: UPDATE FORWARD MODEL WITH GROUND TRUTH DATA PAIR
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            # Save Gradient
            grad_vec_new = torch.cat(
                tuple(
                    torch.squeeze(
                        torch.reshape(par.grad.detach().clone(), (1, torch.numel(par)))
                    )
                    for par in list(model.parameters())
                )
            )
            loss_for_logger += loss
            optimizer.step()
            lr_scheduler.step()

            if BACK and args.train_mode != "nometa":
                # STEP 4: REINFORCE BACKWARD MODEL
                back_optimizer.zero_grad()
                reward = cos(grad_vec_new, grad_vec_old)
                reward = torch.nan_to_num(
                    reward, nan=0.0
                )  # Happens for some reason in first iteration?
                advantage = reward - reward_mv_avg
                reward_mv_avg += advantage / (
                    completed_steps + 1
                )  # Does that work with Accelerate just like that? No race conditions?
                if args.train_mode == "metahard":
                    softmax_probs = torch.clamp(
                        softmax_probs, min=1e-9, max=1 - 1e-9
                    )  # Max was unnecessary. Min is important due to 0*log(0) of entropy calculation
                if args.train_mode == "metadist":
                    softmax_probs = torch.nn.functional.softmax(backward_logits, dim=-1)
                # Average of Sum of Entropies of Vocabulary distributions of each Token
                reinforce = max_ent - torch.mean(
                    torch.sum(
                        torch.special.entr(
                            softmax_probs.view(-1, vocab_size)  # Dissolve Batches
                        ),
                        -1,
                    )
                )
                normalized_reinforce = reinforce / max_ent

                loss = normalized_reinforce * -1 * advantage
                # loss = torch.clamp(loss, max=0)    #Positive only Reinforce (negative loss only)
                accelerator.backward(loss)
                back_loss_for_logger += loss
                back_optimizer.step()
                back_lr_scheduler.step()

                progress_bar.update(1)
                completed_steps += 1

                # Check Backmodel Output occasionally
                if completed_steps % 500 == 0 or (
                    completed_steps % 40 == 0 and completed_steps < 200
                ):
                    back_model.eval()
                    with torch.no_grad():
                        generated_tokens = accelerator.unwrap_model(
                            back_model
                        ).generate(
                            mono_batch["input_ids"],
                            attention_mask=mono_batch["attention_mask"],
                            **back_gen_kwargs,
                        )
                        generated_tokens = accelerator.pad_across_processes(
                            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                        )
                        generated_tokens = (
                            accelerator.gather(generated_tokens).cpu().numpy()
                        )

                        if isinstance(generated_tokens, tuple):
                            generated_tokens = generated_tokens[0]

                        # Replace -100 in the labels as we can't decode them.
                        generated_tokens = np.where(
                            generated_tokens != -100,
                            generated_tokens,
                            tokenizer.pad_token_id,
                        )
                        decoded_preds = tokenizer.batch_decode(
                            generated_tokens, skip_special_tokens=True
                        )

                        decoded_inputs = tokenizer.batch_decode(
                            mono_batch["input_ids"], skip_special_tokens=True
                        )

                        logger.info("SUMMARY:\n”" + str(decoded_inputs[0]) + "\n")
                        logger.info("GENERATED DOC:\n" + str(decoded_preds[0]) + "\n")
                    back_model.train()

            if completed_steps % 20 == 0:
                logger.info(
                    f" Step: {completed_steps}; Loss (Ground Truth): {loss_for_logger/20}; Loss (Generated): {loss_on_generated_for_logger/20}; Loss (Back Model): {back_loss_for_logger/20}"
                )
                loss_for_logger = 0
                loss_on_generated_for_logger = 0
                back_loss_for_logger = 0

        logger.info("VALIDATION:\n")

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():

                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
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
                decoded_labels = tokenizer.batch_decode(
                    labels, skip_special_tokens=True
                )

                decoded_preds, decoded_labels = postprocess_text(
                    decoded_preds, decoded_labels
                )

                metric.add_batch(predictions=decoded_preds, references=decoded_labels)

        result = metric.compute(use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        result = {k: round(v, 4) for k, v in result.items()}

        logger.info(result)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)

    if args.back_model_output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(back_model)
        unwrapped_model.save_pretrained(
            args.back_model_output_dir, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.back_model_output_dir)


if __name__ == "__main__":
    main()
