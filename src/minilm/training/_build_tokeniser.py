from src.minilm.tokeniser import train_tokeniser, load_tokeniser


def get_tokeniser():
    try:
        return load_tokeniser("tokeniser.json")
    except FileNotFoundError:
        return train_tokeniser(
            "data/smollm-corpus-dev/cosmopedia-v2/*.parquet",
            "data/smollm-corpus-dev/fineweb-edu-dedup/*.parquet",
            outfile="tokeniser.json",
        )


if __name__ == "__main__":
    get_tokeniser()
