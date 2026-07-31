from src.minilm.model.model import MiniLM
from src.minilm.training._build_tokeniser import get_tokeniser
from loguru import logger
from src.minilm.training.schedule import get_lr_scheduler
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from src.minilm.training.dataloader import StreamingDataset
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from src.minilm.training.test import run_test_prompts
import typer


MICRO_BATCH_SIZE = 32
EFFECTIVE_BATCH_SIZE = 128
ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // MICRO_BATCH_SIZE
SEQUENCE_LENGTH = 2048
TOKENS_PER_FULL_BATCH = SEQUENCE_LENGTH * EFFECTIVE_BATCH_SIZE
TOKENS_PER_MICRO_BATCH = SEQUENCE_LENGTH * MICRO_BATCH_SIZE


def main(run_name, max_steps: int = 10_000):
    device = "cuda"

    tokeniser = get_tokeniser()

    model = MiniLM(
        vocab_size=tokeniser.get_vocab_size(),
        d_model=64 * 4,
        n_layers=6,
        padding_idx=tokeniser.token_to_id("<|pad|>"),
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Initialised model with {num_params:,} params.")

    dataset = StreamingDataset(tokeniser)

    opt = torch.optim.AdamW(model.parameters(), fused=True, lr=3e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lr_lambda=get_lr_scheduler(
            start_factor=0.01,
            end_factor=0.1,
            warmup_steps=max_steps * 0.2,
            constant_steps=max_steps * 0.1,
            decay_steps=max_steps * 0.5,
        ),
    )

    dataloader = DataLoader(
        dataset, batch_size=MICRO_BATCH_SIZE, num_workers=1, pin_memory=True
    )

    writer = SummaryWriter(log_dir=f"runs/{run_name}")
    writer.add_hparams(
        {
            "run_name": run_name,
            "max_steps": max_steps,
            "num_params": num_params,
            "vocab_size": tokeniser.get_vocab_size(),
            "d_model": 64 * 4,
            "n_layers": 6,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "accumulation_steps": ACCUMULATION_STEPS,
            "sequence_length": SEQUENCE_LENGTH,
            "tokens_per_micro_batch": TOKENS_PER_MICRO_BATCH,
            "tokens_per_full_batch": TOKENS_PER_FULL_BATCH,
            "optimizer": "AdamW",
            "learning_rate": 3e-4,
            "scheduler_start_factor": 0.01,
            "scheduler_end_factor": 0.1,
            "scheduler_warmup_steps": max_steps * 0.2,
            "scheduler_constant_steps": max_steps * 0.1,
            "scheduler_decay_steps": max_steps * 0.5,
        },
        {"model/num_params": num_params},
    )
    pbar = tqdm(total=max_steps)

    step = 0
    while step < max_steps:
        model.zero_grad()
        accum_loss = 0

        # for i, (x, y, mask) in enumerate(dataloader):
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # mask = mask.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda"):
                y_pred = model(x)
                loss = F.cross_entropy(y_pred.transpose(1, 2), y) / ACCUMULATION_STEPS
                accum_loss += loss.item()

            loss.backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                opt.step()
                opt.zero_grad()
                step += 1
                pbar.update()
                writer.add_scalar("Loss", accum_loss, step)
                accum_loss = 0

                scheduler.step()
                writer.add_scalar("LR", opt.param_groups[0]["lr"], step)

            if step % 100 == 0:
                generations = run_test_prompts(model, tokeniser)
                for prompt, completion in generations:
                    writer.add_text(prompt, completion, global_step=step)

            if step > max_steps:
                break

    pbar.close()
    writer.close()


if __name__ == "__main__":
    typer.run(main)
# uv run python -m src.minilm.training.train
