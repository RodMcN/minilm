import sys

import torch
import torch.nn.functional as F
import typer
from loguru import logger
from torch.utils.data import DataLoader
from src.minilm.training.metrics_writer import MetricsWriter
from tqdm.auto import tqdm
import os

from src.minilm.model.model import MiniLM
from src.minilm.training._build_tokeniser import get_tokeniser
from src.minilm.training.dataloader import (
    StreamingDataset,
    SimpleDataset,
)
from src.minilm.training.test import run_test_prompts
from src.minilm.training.optim import get_lr_scheduler, configure_optimiser
from src.minilm.tokeniser import load_tokeniser

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
from contextlib import nullcontext
from src.minilm.training.config import Config
from src.minilm.training.seed import seed_everything, worker_init_fn


def setup_dist():
    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    return local_rank


def cleanup_dist():
    dist.destroy_process_group()


def main(run_name, config):
    local_rank = setup_dist()
    global_rank = dist.get_rank()
    group_rank = os.environ.get("NODE_RANK")
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    world_size = int(os.environ["WORLD_SIZE"])

    config = Config.from_yaml(config)
    run_config = config.run

    micro_batch_size = run_config.micro_batch_size
    effective_batch_size = run_config.effective_batch_size

    local_accumulation_steps = effective_batch_size // world_size // micro_batch_size
    assert local_accumulation_steps > 0
    if global_rank == 0:
        logger.info(
            f"{effective_batch_size=}, {world_size=}, {micro_batch_size=}. {local_accumulation_steps=}"
        )

    # same seed before model init
    # re-seed again after init with rank
    seed_everything(run_config.seed, rank=0)

    tokeniser = load_tokeniser("tokeniser.json")
    pad_id = tokeniser.token_to_id("<|pad|>")

    model_kwargs = {
        "vocab_size": tokeniser.get_vocab_size(),
        "padding_idx": pad_id,
        **config.model.model_dump(),
    }
    model = MiniLM(**model_kwargs).to(device)

    model = DDP(model, device_ids=[local_rank])
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Initialised model with {num_params:,} params.")

    # seed again after init
    rank_seed = seed_everything(run_config.seed, rank=global_rank)

    dataset = SimpleDataset(
        tokeniser,
        sequence_length=run_config.sequence_length,
        rank=global_rank,
        world_size=world_size,
    )

    loader_gen = torch.Generator()
    loader_gen.manual_seed(rank_seed)
    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=loader_gen,
    )

    opt = configure_optimiser(model.parameters(), lr=run_config.lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lr_lambda=get_lr_scheduler(
            start_factor=0.01,
            end_factor=0.1,
            warmup_steps=run_config.max_steps * 0.1,
            constant_steps=0,
            decay_steps=run_config.max_steps * 0.9,
        ),
    )

    if global_rank == 0:
        writer = MetricsWriter(run_name)
        pbar = tqdm(
            total=run_config.max_steps,
            disable=(not sys.stdout.isatty()),
            dynamic_ncols=True,
        )
    else:
        writer = None
        pbar = None

    if writer:
        writer.add_hparams(
            {"run_name": run_name, **config.model_dump()},
            {"model/num_params": num_params},
        )

    step = 0
    while step < run_config.max_steps:
        model.zero_grad()
        accum_ce_loss = 0

        # pad_id

        for i, batch in enumerate(dataloader):
            _is_sync_batch = (i + 1) % local_accumulation_steps == 0
            # logger.debug(
            #     f"Rank [{group_rank},{local_rank}] on batch {i:,}. Synching: {_is_sync_batch}"
            # )
            sync_context = nullcontext if _is_sync_batch else model.no_sync

            if len(batch) == 3:
                x, y, mask = batch
            else:
                x, y, mask = *batch, None

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if mask is not None:
                mask = mask.to(device, non_blocking=True)
            with sync_context():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    valid = y != -100
                    n_total = valid.sum()
                    y_pred = model(x, attn_mask=mask).float()

                    z = torch.logsumexp(y_pred, dim=-1)

                    safe_y = y.masked_fill(~valid, 0)
                    target_logits = y_pred.gather(
                        dim=-1,
                        index=safe_y.unsqueeze(-1),
                    ).squeeze(-1)

                    nll = z - target_logits

                    ce_loss = torch.where(valid, nll, 0.0).sum()
                    z_loss = torch.where(valid, z.square(), 0.0).sum()

                    loss = (
                        (ce_loss + 1e-4 * z_loss) / n_total / local_accumulation_steps
                    )

                    accum_ce_loss += (
                        ce_loss.detach() / local_accumulation_steps / n_total
                    )

                loss.backward()

            if _is_sync_batch:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                opt.step()
                opt.zero_grad()

                dist.all_reduce(accum_ce_loss, dist.ReduceOp.SUM)
                accum_ce_loss /= world_size

                step += 1
                if pbar:
                    pbar.update()
                if writer:
                    writer.add_scalar(
                        "train/ce_loss",
                        accum_ce_loss,
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
                accum_ce_loss = 0

                scheduler.step()
                if step % 1000 == 0:
                    if global_rank == 0:
                        generations = run_test_prompts(model.module, tokeniser)
                        for prompt, completion, completion_tokens in generations:
                            writer.add_text(
                                prompt,
                                completion,
                                global_step=step,
                            )
                dist.barrier()

            if step >= run_config.max_steps:
                break

    if pbar:
        pbar.close()
    if writer:
        writer.close()


if __name__ == "__main__":
    try:
        typer.run(main)
    except Exception as e:
        logger.exception(e)
        logger.complete()
        sys.stderr.flush()
    finally:
        if dist.is_initialized():
            cleanup_dist()
# uv run python -m src.minilm.training.train
