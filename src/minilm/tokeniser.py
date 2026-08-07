import string
from typing import Final
from datasets import load_dataset, interleave_datasets
import tokenizers
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from pathlib import Path


PAD_TOKEN: Final = "<|pad|>"

SYSTEM_TOKEN: Final = "<|system|>"
USER_TOKEN: Final = "<|user|>"
ASSISTANT_TOKEN: Final = "<|assistant|>"
TOOL_TOKEN: Final = "<|tool|>"

END_MESSAGE_TOKEN: Final = "<|end_message|>"

THINK_TOKEN: Final = "<|think|>"
END_THINK_TOKEN: Final = "<|end_think|>"

TOOL_CALL_TOKEN: Final = "<|tool_call|>"
END_TOOL_CALL_TOKEN: Final = "<|end_tool_call|>"

ROLE_TOKENS: Final[dict[str, str]] = {
    "system": SYSTEM_TOKEN,
    "user": USER_TOKEN,
    "assistant": ASSISTANT_TOKEN,
    "tool": TOOL_TOKEN,
}

AA_TOKENS: Final[tuple[str, ...]] = tuple(
    f"<|AA.{letter}|>" for letter in string.ascii_uppercase
)

SPECIAL_TOKENS: Final[tuple[str, ...]] = (
    PAD_TOKEN,
    *ROLE_TOKENS.values(),
    END_MESSAGE_TOKEN,
    THINK_TOKEN,
    END_THINK_TOKEN,
    TOOL_CALL_TOKEN,
    END_TOOL_CALL_TOKEN,
    *AA_TOKENS,
)

DEFAULT_VOCAB_SIZE = 12_000


def train_tokeniser(
    *datasets,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_samples: int | None = None,
    outfile: str | Path | None = None,
):
    combined_datasets = interleave_datasets(
        [
            load_dataset(
                "parquet", data_files=data_files, split="train", streaming=True
            ).select_columns("text")
            for data_files in datasets
        ]
    )

    def batch_iterator(dataset, text_column="text", batch_size=1000):
        batch = []

        for i, example in enumerate(dataset):
            if max_samples is not None and i >= max_samples:
                break

            text = example.get(text_column)

            batch.append(text)

            if len(batch) == batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    tokeniser = Tokenizer(models.BPE())
    tokeniser.normalizer = normalizers.NFKC()
    tokeniser.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokeniser.decoder = tokenizers.decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(SPECIAL_TOKENS),
    )

    tokeniser.train_from_iterator(batch_iterator(combined_datasets), trainer=trainer)

    if outfile:
        save_tokeniser(tokeniser, outfile)
    return tokeniser


def load_tokeniser(path):
    if Path(path).exists():
        return Tokenizer.from_file(path)
    else:
        raise FileNotFoundError(f"{path!r} does not exist.")


def save_tokeniser(tokeniser, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    tokeniser.save(filepath)
