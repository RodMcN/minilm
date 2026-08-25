from pathlib import Path

import torch
from pydantic import BaseModel
from tokenizers import Tokenizer
from torch.nn.parallel import DistributedDataParallel as DDP

from src.minilm.model.model import MiniLM


def save_model(model: torch.nn.Module | DDP, config, tokeniser: Tokenizer, out_path):
    if isinstance(model, DDP):
        model = model.module
    if isinstance(config, BaseModel):
        config = config.model_dump()

    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    torch.save(
        {"config": config, "tokeniser": tokeniser.to_str(), "weights": state_dict},
        out_path,
    )


def load_model(path, device=None):
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")

    model = MiniLM(**checkpoint["config"])
    model.load_state_dict(checkpoint["weights"])
    if device is not None:
        model = model.to(device)

    tokeniser = Tokenizer.from_str(checkpoint["tokeniser"])

    return model, tokeniser
