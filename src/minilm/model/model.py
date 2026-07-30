from torch import nn
from .modules import TransformerStack
import yaml
from pathlib import Path
from os import PathLike


class Model(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, padding_idx, tied=True):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.transformer = TransformerStack(
            n_layers=n_layers,
            d_model=d_model,
        )

        self.norm_out = nn.RMSNorm(d_model)
        self.linear_out = nn.Linear(d_model, vocab_size, bias=False)

        if tied:
            self.linear_out.weight = self.emb.weight

    def forward(self, token_ids, attn_mask=None):
        x = self.emb(token_ids)
        x = self.transformer(x, attn_mask=attn_mask)
        x = self.norm_out(x)
        x = self.linear_out(x)
        return x

    @staticmethod
    def from_config(config: str | PathLike[str] | dict):
        if not isinstance(config, dict):
            path = Path(config)
            if not Path.is_file():
                raise FileNotFoundError(f"{config:!r} is not a file")

            with path.open("r", encoding="utf-8") as file:
                config = yaml.safe_load(file)

        return Model(**config)
