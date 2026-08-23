from torch import nn
from .modules import TransformerStack
import yaml
from pathlib import Path
from os import PathLike
import torch


class MiniLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        n_layers,
        n_head,
        padding_idx,
        tied=True,
        qk_norm=True,
    ):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.transformer = TransformerStack(
            n_layers=n_layers,
            d_model=d_model,
            nhead=n_head,
            qk_norm=qk_norm,
        )

        self.norm_out = nn.RMSNorm(d_model)
        self.linear_out = nn.Linear(d_model, vocab_size, bias=False)

        self.init_weights()

        if tied:
            self.linear_out.weight = self.emb.weight
            with torch.no_grad():
                self.emb.weight[padding_idx].zero_()

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
            if not path.is_file():
                raise FileNotFoundError(f"{config:!r} is not a file")

            with path.open("r", encoding="utf-8") as file:
                config = yaml.safe_load(file)

        return MiniLM(**config)

    def init_weights(self):
        def _init_normal(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_init_normal)

        scale = (2 * len(self.transformer.layers)) ** -0.5
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and name.endswith(
                ("attn_out_proj", "down_proj")
            ):
                module.weight.data.mul_(scale)
