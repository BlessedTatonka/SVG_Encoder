import argparse
import gc
import logging
import os
import random
from functools import partial

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

RANDOM_SEED = 42

# Metadata for the Russian SuperGLUE tasks
svg_superglue_tasks = {
    "sgp-bench": {
        "abbr": "sgp-bench",
        "name": "SGP-bench",
        "metrics": "F1/Accuracy",
        "dataset_names": {"train": "train", "valid": "val", "test": "test"},
        "inputs": ["Question", "A", "B", "C", "D"],
        "target": "Answer",
        "metric_funcs": [accuracy_score, f1_score],
        "n_labels": 4,
    }
}


def fix_seed(random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    set_seed(random_seed)


def compute_metrics(eval_pred, task_metrics):
    predictions, labels = eval_pred

    metrics_d = {}
    for metric_func in task_metrics:
        metric_name = metric_func.__name__
        if metric_name in ["pearsonr", "spearmanr"]:
            score = metric_func(labels, np.squeeze(predictions))
        elif metric_name in ["f1_score"]:
            score = metric_func(np.argmax(predictions, axis=-1), labels, average="macro")
        else:
            score = metric_func(np.argmax(predictions, axis=-1), labels)

        if isinstance(score, tuple):
            metrics_d[metric_func.__name__] = score[0]
        else:
            metrics_d[metric_func.__name__] = score

    return metrics_d


class MetricsCallback(TrainerCallback):
    """Callback to store train and eval metrics at each logging step."""

    def __init__(self):
        self.training_history = {"train": [], "eval": []}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:  # Training logs
                self.training_history["train"].append(logs)
            elif "eval_loss" in logs:  # Evaluation logs
                self.training_history["eval"].append(logs)


def main():
    # --- Parse command-line arguments ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="")
    parser.add_argument("--model_revision", type=str, required=False, default=None, help="")
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        choices=list(svg_superglue_tasks.keys()),
        help="Task name from Russian SuperGLUE.",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for fine-tuning.")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="")
    args = parser.parse_args()

    # Fix random seed for reproducibility
    fix_seed(RANDOM_SEED)

    # Prepare run name
    run_name = (
        f"model_name-{args.model_name}_"
        f"task-{args.task_name}_"
        f"lr-{args.learning_rate}_"
        f"epochs-{args.num_train_epochs}_"
        f"wd-{args.weight_decay}_"
        f"bsz-{args.batch_size}_"
        f"sch-{args.lr_scheduler_type}"
    )

    # Setup logger
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logging.info(f"Starting run: {run_name}")

    # ----- Model and hyperparameters -----
    model_name = args.model_name
    model_revision = args.model_revision

    eps = 1e-6
    betas = (0.9, 0.98)
    train_bsz, val_bsz = args.batch_size, 64
    lr = args.learning_rate
    n_epochs = args.num_train_epochs
    wd = args.weight_decay
    task = args.task_name
    lr_scheduler_type = args.lr_scheduler_type

    # --- Task metadata ---
    task_meta = svg_superglue_tasks[task]
    train_ds_name = task_meta["dataset_names"]["train"]
    valid_ds_name = task_meta["dataset_names"]["valid"]
    test_ds_name = task_meta["dataset_names"]["test"]

    task_inputs = task_meta["inputs"]
    task_target = task_meta["target"]
    n_labels = task_meta["n_labels"]
    task_metrics = task_meta["metric_funcs"]

    save_path = f"results_{model_name.replace('/', '_')}_{model_revision}"
    os.makedirs(save_path, exist_ok=True)
    csv_filename = f"{task}_ft_lr={lr}_n_epochs={n_epochs}_wd={wd}_bsz={train_bsz}_scheduler={lr_scheduler_type}.csv"

    if os.path.exists(os.path.join(save_path, csv_filename)):
        print("ALREADY TRAINED FOR THIS ARGS")
        return

    # --- Load data ---
    logging.info(f"Loading dataset for task: {task}")
    raw_datasets = load_dataset("VectorGraphics/svg-super-glue", task)



    id2label, label2id = {0: "A", 1: "B", 2: "C", 3: "D"}, {"A": 0, "B": 1, "C": 2, "D": 3}

    logging.info("Loading tokenizer and model...")
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        revision=model_revision,
        num_labels=n_labels,
        id2label=id2label,
        label2id=label2id,
        # attn_implementation="sdpa",  # To fix randomness
    ).bfloat16()
    
    hf_data_collator = DataCollatorWithPadding(tokenizer=hf_tokenizer)

    # If you need to do custom tokenization, define a preprocessing function
    def preprocess_function(examples, task_inputs):
        input_sequences = zip(*[examples[inp] for inp in task_inputs])
        texts = [hf_tokenizer.sep_token.join(parts) for parts in input_sequences]
        tokenized = hf_tokenizer(
            texts, 
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
            )
        
        # Add labels to the tokenized output
        if task_target in examples and examples[task_target] is not None:
            tokenized["labels"] = [label2id.get(label, 0) for label in examples[task_target]]
            
        return tokenized

    # For compute_metrics
    task_compute_metrics = partial(compute_metrics, task_metrics=task_metrics)

    # I did that to speed up a bit, but sometimes it raise an exception.
    # tokenized_datasets = load_dataset("TatonkaHF/USER_V2_EVALUATION_RSG_DATASETS", data_dir=task)

    tokenized_datasets = raw_datasets.map(
        partial(preprocess_function, task_inputs=task_inputs), batched=True, batch_size=1000, num_proc=1
    )

    training_args = TrainingArguments(
        output_dir=f"Moderna-small_{task}_ft",
        learning_rate=lr,
        num_train_epochs=n_epochs,
        weight_decay=wd,
        per_device_train_batch_size=train_bsz,
        per_device_eval_batch_size=val_bsz,
        lr_scheduler_type=lr_scheduler_type,
        optim="adamw_torch",
        adam_beta1=betas[0],
        adam_beta2=betas[1],
        adam_epsilon=eps,
        logging_strategy="epoch",
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="no",
        save_steps=1_000_000,
        fp16=False,
        bf16=False,
        bf16_full_eval=False,
        push_to_hub=False,
        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,
        dataloader_num_workers=6,
        warmup_ratio=0.2,
        report_to=[],
    )

    # Prepare Trainer
    trainer = Trainer(
        model=hf_model,
        args=training_args,
        train_dataset=tokenized_datasets[train_ds_name],
        eval_dataset=tokenized_datasets[valid_ds_name],
        tokenizer=hf_tokenizer,
        data_collator=hf_data_collator,
        compute_metrics=task_compute_metrics,
    )

    # Callback to store metrics
    metrics_callback = MetricsCallback()
    trainer.add_callback(metrics_callback)

    logging.info("Starting training...")
    trainer.train()

    logging.info("Training completed. Collecting metrics and saving results...")
    train_history_df = pd.DataFrame(metrics_callback.training_history["train"]).add_prefix("train_")
    eval_history_df = pd.DataFrame(metrics_callback.training_history["eval"]).add_prefix("eval_")

    # Optionally combine them
    combined_df = pd.concat([train_history_df, eval_history_df], axis=1)

    # Save results
    combined_df.to_csv(os.path.join(save_path, csv_filename), index=False)
    logging.info(f"Saved training log to {os.path.join(save_path, csv_filename)}")


if __name__ == "__main__":
    main()