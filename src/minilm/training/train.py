import sys

import torch
import torch.nn.functional as F
import typer
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import os

from src.minilm.model.model import MiniLM
from src.minilm.training._build_tokeniser import get_tokeniser
from src.minilm.training.dataloader import (
    StreamingDataset,
    SimpleDataset,
)
from src.minilm.training.test import run_test_prompts
from src.minilm.training.schedule import get_lr_scheduler
from src.minilm.tokeniser import load_tokeniser

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

MICRO_BATCH_SIZE = 32
EFFECTIVE_BATCH_SIZE = 128
SEQUENCE_LENGTH = 512
TOKENS_PER_FULL_BATCH = SEQUENCE_LENGTH * EFFECTIVE_BATCH_SIZE
TOKENS_PER_MICRO_BATCH = SEQUENCE_LENGTH * MICRO_BATCH_SIZE


LR = 5e-4
# LR = 5e-5

D_MODEL = 64 * 4
N_LAYERS = 10
N_HEAD = 4
# OPT = "muon"


def setup_dist():
    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    return local_rank


def cleanup_dist():
    dist.destroy_process_group()


def main(run_name, max_steps: int = 50_000):
    local_rank = setup_dist()
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)

    world_size = int(os.environ["WORLD_SIZE"])
    local_accumulation_steps = EFFECTIVE_BATCH_SIZE // world_size // MICRO_BATCH_SIZE
    assert local_accumulation_steps > 0
    if global_rank == 0:
        logger.info(
            f"{EFFECTIVE_BATCH_SIZE=}, {world_size=}, {MICRO_BATCH_SIZE=}. {local_accumulation_steps=}"
        )

    tokeniser = load_tokeniser("tokeniser.json")
    pad_id = tokeniser.token_to_id("<|pad|>")

    model = MiniLM(
        vocab_size=tokeniser.get_vocab_size(),
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_head=N_HEAD,
        padding_idx=pad_id,
        tied=False,
    ).to(device)

    model = DDP(model, device_ids=[local_rank])

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Initialised model with {num_params:,} params.")

    # dataset = StreamingDataset(tokeniser)
    dataset = SimpleDataset(tokeniser, sequence_length=SEQUENCE_LENGTH)
    dataloader = DataLoader(
        dataset, batch_size=MICRO_BATCH_SIZE, num_workers=2, pin_memory=True
    )

    # sampler = DistributedSampler(dataset, shuffle=True)

    # muon_params = []
    # adam_params = []

    # for name, p in model.named_parameters():
    #     if not p.requires_grad:
    #         continue

    #     is_muon_candidate = (
    #         p.ndim == 2
    #         and "emb" not in name.lower()
    #         and "linear_out" not in name.lower()
    #     )

    #     if is_muon_candidate:
    #         muon_params.append(p)
    #     else:
    #         adam_params.append(p)

    # muon = torch.optim.Muon(
    #     muon_params,
    #     lr=0.02,
    # )

    # adam = torch.optim.AdamW(adam_params, lr=3e-4, fused=True, weight_decay=0.01)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True, weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lr_lambda=get_lr_scheduler(
            start_factor=0.01,
            end_factor=0.1,
            warmup_steps=max_steps * 0.1,
            constant_steps=max_steps * 0.05,
            decay_steps=max_steps * 0.7,
        ),
    )

    run_config = {
        "max_steps": max_steps,
        "num_params": num_params,
        "vocab_size": tokeniser.get_vocab_size(),
        "d_model": 64 * 4,
        "n_layers": 6,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "tokens_per_micro_batch": TOKENS_PER_MICRO_BATCH,
        "tokens_per_full_batch": TOKENS_PER_FULL_BATCH,
        "optimizer": "AdamW",
        "learning_rate": LR,
    }

    if global_rank == 0:
        writer = SummaryWriter(log_dir=f"runs/{run_name}")
    else:
        writer = None

    if writer:
        writer.add_hparams(
            {"run_name": run_name, **run_config},
            {"model/num_params": num_params},
        )
        pbar = tqdm(
            total=max_steps,
            disable=(not sys.stdout.isatty()) and global_rank == 0,
            dynamic_ncols=True,
        )

    step = 0
    while step < max_steps:
        model.zero_grad()
        accum_loss = 0

        # pad_id

        for i, batch in enumerate(dataloader):
            # TODO: no_sync

            if len(batch) == 3:
                x, y, mask = batch
            else:
                x, y, mask = *batch, None

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if mask is not None:
                mask = mask.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                valid = y != -100
                n_total = valid.sum()
                y_pred = model(x, attn_mask=mask)

                ce_loss = F.cross_entropy(y_pred.transpose(1, 2), y, reduction="sum")
                z = torch.logsumexp(y_pred.float(), dim=-1)
                loss = (
                    (ce_loss + 1e-4 * z[valid].square().sum())
                    / n_total
                    / local_accumulation_steps
                )

                accum_loss += ce_loss.item() / local_accumulation_steps / n_total

            loss.backward()

            if (i + 1) % local_accumulation_steps == 0:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                opt.step()
                opt.zero_grad()
                # muon.step()
                # adam.step()

                # muon.zero_grad()
                # adam.zero_grad()

                step += 1
                pbar.update()
                writer.add_scalar(
                    "train/loss",
                    accum_loss,
                    global_step=step,
                )
                writer.add_scalar(
                    "train/lr",
                    opt.param_groups[0]["lr"],
                    global_step=step,
                )
                writer.add_scalar(
                    "train/grad_norm",
                    total_norm,
                    global_step=step,
                )
                accum_loss = 0

                scheduler.step()
                if step % 1000 == 0:
                    generations = run_test_prompts(model, tokeniser)
                    for prompt, completion, completion_tokens in generations:
                        writer.add_text(
                            prompt,
                            completion,
                            global_step=step,
                        )

            if step >= max_steps:
                break

    pbar.close()
    writer.close()
    cleanup_dist()


if __name__ == "__main__":
    typer.run(main)
# uv run python -m src.minilm.training.train
