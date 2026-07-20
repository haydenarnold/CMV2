#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run predict_moralbert_sft.py for every year from start to end inclusive."
        )
    )
    parser.add_argument(
        "--script-path",
        type=str,
        default="predict_moralbert_sft.py",
        help="Path to the prediction script.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2013,
        help="First year to run.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2024,
        help="Last year to run.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/mnt/hdd/juheechoi/dataset_by_year",
        help="Data directory containing dataset_YEAR.feather files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for scored datasets.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./saved_models",
        help="Directory containing the 8 trained model folders.",
    )
    parser.add_argument(
        "--text-batch-size",
        type=int,
        default=64,
        help="Text batch size for inference.",
    )
    parser.add_argument(
        "--chunk-batch-size",
        type=int,
        default=64,
        help="Chunk batch size for inference.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=150,
        help="Max token length for each chunk.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=30,
        help="Stride for long sentences.",
    )
    parser.add_argument(
        "--sentence-overlap",
        type=int,
        default=1,
        help="Sentence overlap between chunks.",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Enable AMP if supported.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing scored output and checkpoint files.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional sample size to test on a smaller subset.",
    )
    return parser.parse_args()


def run_year(
    script_path: Path,
    year: int,
    data_dir: str,
    output_dir: str,
    models_dir: str,
    text_batch_size: int,
    chunk_batch_size: int,
    max_length: int,
    stride: int,
    sentence_overlap: int,
    use_amp: bool,
    overwrite: bool,
    sample_size: int | None,
) -> None:
    command = [
        sys.executable,
        str(script_path),
        "--year",
        str(year),
        "--data-dir",
        data_dir,
        "--output-dir",
        output_dir,
        "--models-dir",
        models_dir,
        "--text-batch-size",
        str(text_batch_size),
        "--chunk-batch-size",
        str(chunk_batch_size),
        "--max-length",
        str(max_length),
        "--stride",
        str(stride),
        "--sentence-overlap",
        str(sentence_overlap),
    ]

    if use_amp:
        command.append("--use-amp")

    if overwrite:
        command.append("--overwrite")

    if sample_size is not None:
        command.extend(["--sample-size", str(sample_size)])

    print("\n" + "#" * 80)
    print(f"Running prediction for year {year}")
    print("Command:", " ".join(command))
    print("#" * 80 + "\n")

    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    script_path = Path(args.script_path).expanduser().resolve()

    if not script_path.exists():
        raise FileNotFoundError(
            f"Prediction script not found: {script_path}"
        )

    if args.start_year > args.end_year:
        raise ValueError(
            "start-year must be less than or equal to end-year"
        )

    for year in range(args.start_year, args.end_year + 1):
        run_year(
            script_path=script_path,
            year=year,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            models_dir=args.models_dir,
            text_batch_size=args.text_batch_size,
            chunk_batch_size=args.chunk_batch_size,
            max_length=args.max_length,
            stride=args.stride,
            sentence_overlap=args.sentence_overlap,
            use_amp=args.use_amp,
            overwrite=args.overwrite,
            sample_size=args.sample_size,
        )

    print("\nAll years completed successfully.")


if __name__ == "__main__":
    main()
