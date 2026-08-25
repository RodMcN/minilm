import random
from typing import Callable, Iterator, List, Optional

import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from datasets import load_dataset, interleave_datasets

from torch.nn.utils.rnn import pad_sequence
from src.minilm.tokeniser import END_MESSAGE_TOKEN


class SimpleDataset(IterableDataset):
    def __init__(
        self,
        tokeniser,
        sequence_length=2048,
        tokenise_batch_size=500,
        rank=0,
        world_size=1,
    ):
        super().__init__()

        self.tokeniser = tokeniser
        self.sequence_length = sequence_length
        self.tokenise_batch_size = tokenise_batch_size
        self.rank = rank
        self.world_size = world_size

        self.dataset = load_dataset(
            "parquet", data_files="data/train/*.parquet", split="train"
        )

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        shard = self.dataset.shard(
            num_shards=self.world_size * num_workers,
            index=self.rank * num_workers + worker_id,
        )
        eos_id = self.tokeniser.token_to_id(END_MESSAGE_TOKEN)

        token_buffer = []
        sequence_length = self.sequence_length

        for batch in shard.iter(batch_size=self.tokenise_batch_size):
            for encoding in self.tokeniser.encode_batch_fast(batch["text"]):
                # pack documents
                # separate with EOS tokens and concatenate
                # attention currently leaks between docs
                # fine for now
                token_buffer.extend(encoding.ids)
                token_buffer.append(eos_id)

            consumed = 0
            while len(token_buffer) - consumed > sequence_length:
                tokens = torch.tensor(
                    token_buffer[consumed : consumed + sequence_length + 1],
                    dtype=torch.long,
                )
                yield tokens[:-1], tokens[1:]
                consumed += sequence_length

            if consumed:
                token_buffer = token_buffer[consumed:]
