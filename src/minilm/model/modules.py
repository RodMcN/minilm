import torch
from torch import nn
import torch.nn.functional as F
from inspect import getmembers


def get_module(module_name: str):
    for name, module in getmembers(nn.modules):
        if name == module_name:
            return module
    raise ValueError(f"{module_name!r} is not a nn.Module")


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()

        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        value = self.up_proj(x)
        return self.down_proj(gate * value)


class RoPE(nn.Module):
    """Interleaved RoPE for [batch, heads, sequence, head_dim] tensors."""

    def __init__(self, head_dim: int, base: float = 10_000.0):
        super().__init__()

        if head_dim % 2:
            raise ValueError(f"head_dim must be even, got {head_dim}")

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def rotate_pairs(x: torch.Tensor) -> torch.Tensor:
        # (x0, x1) -> (-x1, x0)
        x = x.reshape(*x.shape[:-1], -1, 2)
        x1, x2 = x.unbind(dim=-1)
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)

        if positions is None:
            # Normal full-sequence training.
            positions = torch.arange(
                seq_len,
                device=q.device,
                dtype=torch.float32,
            )
        else:
            positions = positions.to(
                device=q.device,
                dtype=torch.float32,
            )

        # [seq, head_dim / 2], or [batch, seq, head_dim / 2]
        angles = positions.unsqueeze(-1) * self.inv_freq

        cos = angles.cos().repeat_interleave(2, dim=-1)
        sin = angles.sin().repeat_interleave(2, dim=-1)

        # [seq, dim]       -> [1, 1, seq, dim]
        # [batch, seq, dim] -> [batch, 1, seq, dim]
        while cos.ndim < q.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        cos = cos.to(dtype=q.dtype)
        sin = sin.to(dtype=q.dtype)

        q = q * cos + self.rotate_pairs(q) * sin
        k = k * cos + self.rotate_pairs(k) * sin

        return q, k


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        norm_type: str | nn.Module,
        qk_norm=True,
        rope=True,
    ):
        super().__init__()

        norm_type = (
            norm_type if isinstance(norm_type, nn.Module) else get_module(norm_type)
        )

        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.norm1 = norm_type(d_model)
        self.norm2 = norm_type(d_model)

        self.qk_norm = qk_norm
        if self.qk_norm:
            self.q_norm = norm_type(self.head_dim)
            self.k_norm = norm_type(self.head_dim)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out_proj = nn.Linear(d_model, d_model, bias=False)

        self.out = SwiGLU(d_model, dim_feedforward)

        if rope:
            self.rope = RoPE(self.head_dim)
        else:
            self.rope = None

    def forward(self, x, attn_mask=None, is_causal=True):

        batch_size, seq_len, d_model = x.shape

        h = self.norm1(x)
        qkv = self.qkv_proj(h).reshape(
            batch_size,
            seq_len,
            3,
            self.nhead,
            self.head_dim,
        )

        q, k, v = qkv.unbind(dim=2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # [batch, seq len, n head, d head]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # [batch, n head, seq len, d head]

        if self.rope is not None:
            q, k = self.rope(q, k)

        if attn_mask is not None and is_causal:
            causal_mask = torch.ones(
                seq_len, seq_len, dtype=torch.bool, device=x.device
            ).tril()
            attn_mask = attn_mask[:, None, None, :] & causal_mask
            is_causal = False

        h = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal
        )

        h = h.transpose(1, 2).contiguous()
        h = h.reshape(batch_size, seq_len, d_model)

        x = x + self.attn_out_proj(h)

        x = x + self.out(self.norm2(x))

        return x


class TransformerStack(nn.Module):
    def __init__(
        self,
        n_layers,
        d_model,
        nhead: int | None = None,
        dim_feedforward: int | None = None,
        norm_type="RMSNorm",
        qk_norm=True,
    ):
        super().__init__()

        nhead = nhead or d_model // 64
        dim_feedforward = dim_feedforward or d_model * 4

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, nhead, dim_feedforward, norm_type, qk_norm=qk_norm
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, *args, **kwargs):
        for layer in self.layers:
            x = layer(x, *args, **kwargs)
        return x
