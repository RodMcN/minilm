import argparse
import math
from pathlib import Path
from typing import Sequence

from datasets import load_dataset

DATASET_ID = "HuggingFaceTB/smollm-corpus"
SUBSETS = ("fineweb-edu-dedup", "cosmopedia-v2")  # , "python-edu")
DEFAULT_OUTPUT_DIR = Path("data/smollm-corpus-dev")


def download_dataset_fraction(
    dataset_id: str,
    output_dir: Path,
    *,
    configs: Sequence[str] | None = None,
    split: str = "train",
    fraction: float = 0.01,
    max_examples: int | None = None,
) -> None:

    for config in configs if configs is not None else (None,):
        config_name = config or "default"
        output_path = Path(output_dir) / config_name / f"{split}.parquet"

        dataset = load_dataset(
            dataset_id,
            config,
            split=split,
            streaming=True,
        )
        split_info = dataset.info.splits and dataset.info.splits.get(split)
        if split_info is None or split_info.num_examples is None:
            raise ValueError(f"split {split!r} has no size metadata in {dataset_id!r}")
        total = split_info.num_examples

        count = max(1, math.ceil(total * fraction))
        if max_examples is not None:
            count = min(count, max_examples)

        source = dataset_id if config is None else f"{dataset_id}/{config}"
        print(f"Writing {count:,} rows from {source} to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        (
            dataset.shuffle(seed=42, buffer_size=min(10_000, count))
            .take(count)
            .to_parquet(output_path)
        )


def download_smollm_dev(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    fraction: float = 0.01,
    max_examples: int | None = None,
    overwrite: bool = False,
) -> None:
    download_dataset_fraction(
        DATASET_ID,
        output_dir,
        configs=SUBSETS,
        fraction=fraction,
        max_examples=max_examples,
        overwrite=overwrite,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_id", nargs="?", default=DATASET_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--config",
        dest="configs",
        action="append",
        help="dataset config to download; repeat for multiple configs",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--fraction", type=float, default=0.01)
    parser.add_argument("--max-examples", type=int)
    args = parser.parse_args()

    configs = args.configs
    if configs is None and args.dataset_id == DATASET_ID:
        configs = SUBSETS

    download_dataset_fraction(
        args.dataset_id,
        args.output_dir,
        configs=configs,
        split=args.split,
        fraction=args.fraction,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()
