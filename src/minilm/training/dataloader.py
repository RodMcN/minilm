import random
from typing import Callable, Iterator, List, Optional

import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from datasets import load_dataset, interleave_datasets

from torch.nn.utils.rnn import pad_sequence
from src.minilm.tokeniser import END_MESSAGE_TOKEN


class StreamingDataset(IterableDataset):
    def __init__(
        self,
        tokeniser,
        sequence_length: int = 2048,
    ) -> None:
        super().__init__()

        self.tokeniser = tokeniser
        self.sequence_length = sequence_length

        cosmopedia = load_dataset(
            "parquet",
            data_files=("data/smollm-corpus-dev/cosmopedia-v2/*.parquet"),
            split="train",
            streaming=True,
        )

        fineweb = load_dataset(
            "parquet",
            data_files=("data/smollm-corpus-dev/fineweb-edu-dedup/*.parquet"),
            split="train",
            streaming=True,
        )

        self.dataset = interleave_datasets(
            [cosmopedia, fineweb],
            stopping_strategy="all_exhausted_without_replacement",
        )

    def __iter__(self):

        # worker_info = get_worker_info()
        # if worker_info is None:
        #     worker_id, num_workers = 0, 1
        # else:
        #     worker_id, num_workers = worker_info.id, worker_info.num_workers

        batch = []

        for example in self.dataset.shuffle(buffer_size=10_000):
            batch.append(example["text"])

            if len(batch) == 256:
                yield from self._encode_and_concat(batch)
                batch.clear()

    def _encode_and_concat(self, texts):

        eos_token_id = self.tokeniser.token_to_id(END_MESSAGE_TOKEN)
        sequence_length = self.sequence_length

        input_ids = []

        for encoding in self.tokeniser.encode_batch(texts):
            # pack documents
            # separate with EOS tokens and concatenate
            # note: attention currently leaks between docs
            # fine for now
            input_ids.extend(encoding.ids + [eos_token_id])

            if len(input_ids) > sequence_length:
                target_ids = input_ids[1 : sequence_length + 1]
                input_ids = input_ids[:sequence_length]

                yield (
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(target_ids, dtype=torch.long),
                )


# not needed with document packing
def collate_fn(batch):
    xs, ys = zip(*batch)

    x = pad_sequence(
        xs,
        batch_first=True,
        padding_value=0,
    )

    y = pad_sequence(
        ys,
        batch_first=True,
        padding_value=-100,  # ignore index
    )

    valid_tokens = x != 0

    return x, y, valid_tokens
