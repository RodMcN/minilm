from src.minilm.tokeniser import train_tokeniser, load_tokeniser
from pathlib import Path


def get_tokeniser():
    try:
        return load_tokeniser("tokeniser.json")
    except FileNotFoundError:
        return train_tokeniser(
            [str(p) for p in Path("data/train").rglob("*.parquet")],
            outfile="tokeniser.json",
            max_samples=1_000_000,
        )


if __name__ == "__main__":
    get_tokeniser()
