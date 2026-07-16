import argparse
import gc
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import nltk
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from nltk.tokenize import sent_tokenize
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


MODEL_BASE = "bert-base-uncased"

MFT_VALUES = [
    "care",
    "harm",
    "fairness",
    "cheating",
    "loyalty",
    "betrayal",
    "authority",
    "subversion",
    "purity",
    "degradation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run sentence-chunked MoralBERT prediction with batched inference."
        )
    )

    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Dataset year. Example: --year 2023",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory containing dataset_YEAR.feather.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory for output and checkpoint files.",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=150,
        help="Maximum token length for each sentence chunk.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=30,
        help="Token overlap for an overlong single sentence.",
    )

    parser.add_argument(
        "--sentence-overlap",
        type=int,
        default=1,
        help="Number of sentences overlapping between adjacent chunks.",
    )

    parser.add_argument(
        "--text-batch-size",
        type=int,
        default=64,
        help=(
            "Number of original texts prepared together. "
            "This mainly affects CPU/RAM use. Default: 64"
        ),
    )

    parser.add_argument(
        "--chunk-batch-size",
        type=int,
        default=64,
        help=(
            "Number of chunks passed to MoralBERT in one forward call. "
            "Use 32-64 for CPU and 128-512 for GPU. Default: 64"
        ),
    )

    parser.add_argument(
        "--use-amp",
        action="store_true",
        help=(
            "Use FP16/BF16 mixed precision when CUDA is available. "
            "Ignored when running on CPU."
        ),
    )

    parser.add_argument(
        "--start-mft",
        type=str,
        default=None,
        choices=MFT_VALUES,
        help=(
            "Start or resume from a particular moral foundation. "
            "Example: --start-mft fairness"
        ),
    )

    parser.add_argument(
        "--only-mft",
        type=str,
        default=None,
        choices=MFT_VALUES,
        help=(
            "Run only one moral foundation. "
            "Useful for testing or parallel GPU execution."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recalculate prediction columns that already exist.",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help=(
            "Optional number of rows used for a test run. "
            "Example: --sample-size 1000"
        ),
    )

    return parser.parse_args()


def ensure_nltk_resources() -> None:
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]

    for resource_path, download_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"Downloading NLTK resource: {download_name}")
            nltk.download(download_name, quiet=False)


class MoralBERTModel(
    nn.Module,
    PyTorchModelHubMixin,
    pipeline_tag="text-classification",
    license="mit",
):
    def __init__(
        self,
        bert_model: nn.Module,
        moral_label: int = 2,
    ):
        super().__init__()

        self.bert = bert_model
        hidden_size = bert_model.config.hidden_size

        self.invariant_trans = nn.Linear(
            hidden_size,
            hidden_size,
        )

        self.moral_classification = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, moral_label),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids

        outputs = self.bert(**model_inputs)

        # BERT [CLS] representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.invariant_trans(pooled_output)

        return self.moral_classification(pooled_output)


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""

    return str(value).strip()


def split_long_sentence(
    sentence: str,
    tokenizer,
    max_length: int,
    stride: int,
) -> list[str]:
    """
    Split one overlong sentence using a token-level sliding window.
    """
    content_capacity = max_length - 2

    if content_capacity <= 0:
        raise ValueError("max_length must be greater than 2.")

    if stride < 0:
        raise ValueError("stride must be zero or greater.")

    if stride >= content_capacity:
        raise ValueError(
            "stride must be smaller than max_length - 2."
        )

    token_ids = tokenizer(
        sentence,
        add_special_tokens=False,
    )["input_ids"]

    if not token_ids:
        return [""]

    step = content_capacity - stride
    pieces = []
    start = 0

    while start < len(token_ids):
        piece_ids = token_ids[
            start:start + content_capacity
        ]

        piece = tokenizer.decode(
            piece_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        pieces.append(piece)

        if start + content_capacity >= len(token_ids):
            break

        start += step

    return pieces


def chunk_by_sentences(
    text: str,
    tokenizer,
    max_length: int = 150,
    stride: int = 30,
    sentence_overlap: int = 1,
) -> list[str]:
    """
    Split text while preserving sentence boundaries.

    If one sentence exceeds the token capacity, apply a token-level
    sliding-window fallback to that sentence.
    """
    text = normalize_text(text)

    if not text:
        return [""]

    content_capacity = max_length - 2

    if content_capacity <= 0:
        raise ValueError("max_length must be greater than 2.")

    if sentence_overlap < 0:
        raise ValueError(
            "sentence_overlap must be zero or greater."
        )

    sentences = sent_tokenize(text)

    if not sentences:
        return [text]

    sentence_lengths = [
        len(
            tokenizer(
                sentence,
                add_special_tokens=False,
            )["input_ids"]
        )
        for sentence in sentences
    ]

    chunks = []
    number_of_sentences = len(sentences)
    start = 0

    while start < number_of_sentences:
        # One sentence alone exceeds the available capacity.
        if sentence_lengths[start] > content_capacity:
            chunks.extend(
                split_long_sentence(
                    sentence=sentences[start],
                    tokenizer=tokenizer,
                    max_length=max_length,
                    stride=stride,
                )
            )

            start += 1
            continue

        end = start
        total_tokens = 0

        while (
            end < number_of_sentences
            and sentence_lengths[end] <= content_capacity
            and (
                total_tokens + sentence_lengths[end]
                <= content_capacity
            )
        ):
            total_tokens += sentence_lengths[end]
            end += 1

        chunks.append(
            " ".join(sentences[start:end])
        )

        if end >= number_of_sentences:
            break

        # Preserve sentence-level overlap without entering an infinite loop.
        start = max(
            start + 1,
            end - sentence_overlap,
        )

    return chunks or [""]


def get_autocast_context(
    device: torch.device,
    use_amp: bool,
):
    """
    Return an AMP context on CUDA and a no-op context on CPU.
    """
    if device.type != "cuda" or not use_amp:
        return nullcontext()

    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    return torch.autocast(
        device_type="cuda",
        dtype=amp_dtype,
    )


@torch.inference_mode()
def predict_texts_batched(
    texts: Iterable[object],
    model: nn.Module,
    tokenizer,
    device: torch.device,
    text_batch_size: int = 64,
    chunk_batch_size: int = 64,
    max_length: int = 150,
    stride: int = 30,
    sentence_overlap: int = 1,
    use_amp: bool = False,
    description: str = "Prediction",
) -> list[float]:
    """
    Predict many original texts using batched chunk inference.

    Each original text may produce multiple sentence chunks. All chunks
    are processed in model batches. The score of each original text is
    the maximum positive-class probability among its chunks.
    """
    if text_batch_size <= 0:
        raise ValueError("text_batch_size must be greater than zero.")

    if chunk_batch_size <= 0:
        raise ValueError("chunk_batch_size must be greater than zero.")

    text_list = list(texts)
    total_texts = len(text_list)

    if total_texts == 0:
        return []

    model.eval()
    all_scores: list[float] = []

    progress = tqdm(
        range(0, total_texts, text_batch_size),
        desc=description,
        unit="text-batch",
    )

    for text_start in progress:
        text_end = min(
            text_start + text_batch_size,
            total_texts,
        )

        current_texts = text_list[
            text_start:text_end
        ]

        flattened_chunks: list[str] = []
        chunk_owner_indices: list[int] = []

        # Sentence splitting is performed per original text, but model
        # inference is performed on flattened chunk batches.
        for local_text_index, raw_text in enumerate(current_texts):
            chunks = chunk_by_sentences(
                text=normalize_text(raw_text),
                tokenizer=tokenizer,
                max_length=max_length,
                stride=stride,
                sentence_overlap=sentence_overlap,
            )

            flattened_chunks.extend(chunks)

            chunk_owner_indices.extend(
                [local_text_index] * len(chunks)
            )

        # Keep one maximum score for each original text in this batch.
        batch_max_scores = [
            float("-inf")
        ] * len(current_texts)

        number_of_chunks = len(flattened_chunks)

        for chunk_start in range(
            0,
            number_of_chunks,
            chunk_batch_size,
        ):
            chunk_end = min(
                chunk_start + chunk_batch_size,
                number_of_chunks,
            )

            chunk_texts = flattened_chunks[
                chunk_start:chunk_end
            ]

            chunk_owners = chunk_owner_indices[
                chunk_start:chunk_end
            ]

            encoded = tokenizer(
                chunk_texts,
                add_special_tokens=True,
                max_length=max_length,
                truncation=True,

                # Dynamic padding:
                # pad only to the longest sequence in this chunk batch.
                padding=True,

                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(
                    device,
                    non_blocking=(device.type == "cuda"),
                )
                for key, value in encoded.items()
            }

            with get_autocast_context(
                device=device,
                use_amp=use_amp,
            ):
                logits = model(**encoded)

            # Convert to float32 before softmax for numerical stability.
            positive_probabilities = F.softmax(
                logits.float(),
                dim=1,
            )[:, 1]

            positive_probabilities = (
                positive_probabilities
                .detach()
                .cpu()
                .tolist()
            )

            for owner_index, probability in zip(
                chunk_owners,
                positive_probabilities,
            ):
                if probability > batch_max_scores[owner_index]:
                    batch_max_scores[owner_index] = float(
                        probability
                    )

            del encoded
            del logits
            del positive_probabilities

        # Every original text should have at least one chunk.
        batch_max_scores = [
            0.0 if score == float("-inf") else score
            for score in batch_max_scores
        ]

        all_scores.extend(batch_max_scores)

        progress.set_postfix(
            {
                "texts": text_end,
                "chunks": number_of_chunks,
                "device": device.type,
            }
        )

    return all_scores


def validate_columns(
    df: pd.DataFrame,
    required_columns: list[str],
) -> None:
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            "Required columns are missing: "
            + ", ".join(missing_columns)
        )


def get_mft_values_to_run(
    start_mft: str | None,
    only_mft: str | None,
) -> list[str]:
    if only_mft is not None:
        return [only_mft]

    if start_mft is None:
        return MFT_VALUES

    start_index = MFT_VALUES.index(start_mft)

    return MFT_VALUES[start_index:]


def build_post_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create one row per post while retaining existing *_post columns
    from a checkpoint.
    """
    existing_post_columns = [
        column
        for column in df.columns
        if column.endswith("_post")
    ]

    selected_columns = [
        "post_id",
        "processed_post",
        *existing_post_columns,
    ]

    # Remove accidental duplicate column names while preserving order.
    selected_columns = list(
        dict.fromkeys(selected_columns)
    )

    return (
        df[selected_columns]
        .drop_duplicates(
            subset="post_id",
            keep="first",
        )
        .copy()
    )


def main() -> None:
    args = parse_args()
    ensure_nltk_resources()

    data_dir = (
        Path(args.data_dir)
        .expanduser()
        .resolve()
    )

    output_dir = (
        Path(args.output_dir)
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_path = (
        data_dir
        / f"dataset_{args.year}.feather"
    )

    output_path = (
        output_dir
        / f"dataset_{args.year}_scored.feather"
    )

    checkpoint_path = (
        output_dir
        / f"dataset_{args.year}_scored_checkpoint.feather"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)
    print(f"Year:              {args.year}")
    print(f"Input:             {input_path}")
    print(f"Output:            {output_path}")
    print(f"Checkpoint:        {checkpoint_path}")
    print(f"Device:            {device}")
    print(f"Text batch size:   {args.text_batch_size}")
    print(f"Chunk batch size:  {args.chunk_batch_size}")
    print(f"Max length:        {args.max_length}")
    print(f"Stride:            {args.stride}")
    print(f"Sentence overlap:  {args.sentence_overlap}")
    print(f"AMP requested:     {args.use_amp}")
    print("=" * 72)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )
    else:
        print(
            "CUDA is unavailable. Batched inference will run on CPU."
        )

    if os.environ.get("HF_TOKEN"):
        print("Hugging Face authentication: HF_TOKEN detected")
    else:
        print(
            "Hugging Face authentication: stored login or public access"
        )

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_BASE
    )

    if checkpoint_path.exists() and not args.overwrite:
        print(
            f"Loading checkpoint: {checkpoint_path}"
        )

        df = pd.read_feather(
            checkpoint_path
        )
    else:
        print(f"Loading input dataset: {input_path}")

        df = pd.read_feather(
            input_path
        )

    validate_columns(
        df,
        [
            "post_id",
            "processed_post",
            "processed_body",
        ],
    )

    if args.sample_size is not None:
        if args.sample_size <= 0:
            raise ValueError(
                "sample_size must be greater than zero."
            )

        df = (
            df.head(args.sample_size)
            .copy()
        )

        print(
            f"Test mode: using the first {len(df):,} rows."
        )

    print(f"Comment rows: {len(df):,}")
    print(
        "Unique posts:",
        f"{df['post_id'].nunique():,}",
    )

    post = build_post_dataframe(df)

    mft_values_to_run = get_mft_values_to_run(
        start_mft=args.start_mft,
        only_mft=args.only_mft,
    )

    for mft in mft_values_to_run:
        comment_column = f"{mft}_comment"
        post_column = f"{mft}_post"

        comment_exists = comment_column in df.columns
        post_exists = post_column in post.columns

        if (
            comment_exists
            and post_exists
            and not args.overwrite
        ):
            print(
                f"\nSkipping {mft}: both prediction columns already exist."
            )
            continue

        print("\n" + "=" * 72)
        print(f"Loading MoralBERT model: {mft}")
        print("=" * 72)

        repo_name = (
            f"vjosap/moralBERT-predict-{mft}-in-text"
        )

        # A fresh base BERT instance is supplied for each MFT model.
        bert_model = AutoModel.from_pretrained(
            MODEL_BASE
        )

        model = MoralBERTModel.from_pretrained(
            repo_name,
            bert_model=bert_model,
        )

        model.to(device)
        model.eval()

        if not comment_exists or args.overwrite:
            print(
                f"Predicting {comment_column} for {len(df):,} comments..."
            )

            df[comment_column] = predict_texts_batched(
                texts=df["processed_body"].tolist(),
                model=model,
                tokenizer=tokenizer,
                device=device,
                text_batch_size=args.text_batch_size,
                chunk_batch_size=args.chunk_batch_size,
                max_length=args.max_length,
                stride=args.stride,
                sentence_overlap=args.sentence_overlap,
                use_amp=args.use_amp,
                description=f"{args.year} {mft} comments",
            )
        else:
            print(
                f"Skipping existing column: {comment_column}"
            )

        if not post_exists or args.overwrite:
            print(
                f"Predicting {post_column} for {len(post):,} posts..."
            )

            post[post_column] = predict_texts_batched(
                texts=post["processed_post"].tolist(),
                model=model,
                tokenizer=tokenizer,
                device=device,
                text_batch_size=args.text_batch_size,
                chunk_batch_size=args.chunk_batch_size,
                max_length=args.max_length,
                stride=args.stride,
                sentence_overlap=args.sentence_overlap,
                use_amp=args.use_amp,
                description=f"{args.year} {mft} posts",
            )
        else:
            print(
                f"Skipping existing column: {post_column}"
            )

        # Replace an existing merged post score when overwrite is enabled.
        if post_column in df.columns:
            df = df.drop(
                columns=[post_column]
            )

        df = df.merge(
            post[["post_id", post_column]],
            how="left",
            on="post_id",
            validate="many_to_one",
        )

        # Save after every completed foundation.
        df.reset_index(drop=True).to_feather(
            checkpoint_path
        )

        print(
            f"Checkpoint saved after {mft}: "
            f"{checkpoint_path}"
        )

        del model
        del bert_model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nSaving final scored dataset...")

    df.reset_index(drop=True).to_feather(
        output_path
    )

    print(f"Completed: {output_path}")

    if (
        checkpoint_path.exists()
        and args.only_mft is None
    ):
        checkpoint_path.unlink()

        print(
            f"Removed completed checkpoint: {checkpoint_path}"
        )


if __name__ == "__main__":
    main()