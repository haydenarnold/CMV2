#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

FOUNDATIONS = [
    "Authority",
    "Care",
    "Equality",
    "Loyalty",
    "Non-Moral",
    "Proportionality",
    "Purity",
    "Thin Morality",
]


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "label": int(self.labels[idx]),
        }


def collate_fn(batch, tokenizer, max_length):
    texts = [item["text"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded["labels"] = labels
    return encoded


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def build_dataset(df, text_col, label_col):
    texts = df[text_col].fillna("").astype(str).tolist()
    labels = df[label_col].fillna(0).astype(int).tolist()
    return TextDataset(texts, labels)


def get_default_device() -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train BERT binary classifiers for each foundation label using 100% of the data."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="mfrc_multilabel.csv",
        help="Path to the CSV dataset file.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="cleaned_text",
        help="Text column to use for training.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="Pretrained BERT model name.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./saved_models",
        help="Directory where each trained model will be saved.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum token length for each example.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for AdamW.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=get_default_device(),
        help="Device to train on (mps, cuda, or cpu).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if args.text_column not in df.columns:
        raise ValueError(f"Text column '{args.text_column}' not found in CSV.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    device = torch.device(args.device)

    for label_name in FOUNDATIONS:
        if label_name not in df.columns:
            raise ValueError(f"Label column '{label_name}' not found in CSV.")

        print(f"\n==== Training binary model for: {label_name} ====\n")

        dataset = build_dataset(df, args.text_column, label_name)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, tokenizer, args.max_length),
            num_workers=2,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=2,
        ).to(device)

        optimizer = AdamW(model.parameters(), lr=args.learning_rate)

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, dataloader, optimizer, device)
            print(f"{label_name} | Epoch {epoch}/{args.epochs} | train_loss: {train_loss:.4f}")

        model_output_dir = output_dir / label_name.replace(" ", "_")
        model_output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_output_dir)
        tokenizer.save_pretrained(model_output_dir)
        print(f"Saved {label_name} model to {model_output_dir}")

    print("\nAll foundation models have been trained and saved.")


if __name__ == "__main__":
    main()
