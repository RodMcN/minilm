import argparse
import math
from pathlib import Path
from typing import Sequence

from datasets import Dataset, load_dataset

DATASET_ID = "HuggingFaceTB/smollm-corpus"
SUBSETS = ("fineweb-edu-dedup", "cosmopedia-v2")  # , "python-edu")
DEFAULT_OUTPUT_DIR = Path("data/smollm-corpus-dev")
ROWS_PER_FILE = 100_000


def download_dataset_fraction(
    dataset_id: str,
    output_dir: Path,
    *,
    configs: Sequence[str] | None = None,
    split: str = "train",
    fraction: float = 0.01,
    max_examples: int | None = None,
    text_col: str = "text",
    data_files: Sequence[str] | None = None,
) -> None:

    for config in configs if configs is not None else (None,):
        config_name = config or "default"
        split_output_dir = Path(output_dir) / config_name

        dataset = load_dataset(
            dataset_id,
            config,
            data_files=data_files,
            split=split,
            streaming=True,
        )
        if text_col != "text":
            dataset = dataset.rename_column(text_col, "text")

        split_info = dataset.info.splits and dataset.info.splits.get(split)
        if split_info is None or split_info.num_examples is None:
            if fraction != 1 or max_examples is None:
                raise ValueError(
                    f"split {split!r} has no size metadata in {dataset_id!r}; "
                    "use fraction=1 with max_examples to request an exact row count"
                )
            count = max_examples
        else:
            count = max(1, math.ceil(split_info.num_examples * fraction))
            if max_examples is not None:
                count = min(count, max_examples)

        source = dataset_id if config is None else f"{dataset_id}/{config}"
        print(f"Writing {count:,} rows from {source} to {split_output_dir}")
        split_output_dir.mkdir(parents=True, exist_ok=True)

        for file_index, batch in enumerate(
            dataset.take(count).iter(batch_size=ROWS_PER_FILE)
        ):
            output_path = split_output_dir / f"{split}-{file_index:05d}.parquet"
            print(f"Writing {len(batch['text']):,} rows to {output_path}")
            Dataset.from_dict(batch, features=dataset.features).to_parquet(output_path)


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
    parser.add_argument(
        "--text-col",
        default="text",
        help="source content column to rename to 'text'",
    )
    parser.add_argument(
        "--data-file",
        dest="data_files",
        action="append",
        help="data file to load directly; repeat for multiple files",
    )
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
        text_col=args.text_col,
        data_files=args.data_files,
    )


if __name__ == "__main__":
    main()
