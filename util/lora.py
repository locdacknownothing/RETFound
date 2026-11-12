import math
from typing import Iterable, Tuple
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper around an existing nn.Linear layer.

    At initialization, lora_B is zero-initialized so the output is bit-for-bit
    identical to the wrapped base layer.
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(base)}")
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r if self.r > 0 else 0.0

        # Freeze original weights and bias
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        if self.r > 0:
            self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
            self.lora_A = nn.Linear(self.in_features, self.r, bias=False)
            self.lora_B = nn.Linear(self.r, self.out_features, bias=False)
            self.reset_lora_parameters()
        else:
            self.lora_dropout = nn.Identity()
            self.lora_A = None
            self.lora_B = None

    def reset_lora_parameters(self):
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.r > 0:
            lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
            out = out + lora_out
        return out


def apply_lora(
    model: nn.Module,
    r: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.05,
    targets: Tuple[str, ...] = ("attn.qkv", "attn.proj"),
) -> int:
    """Replace target Linear submodules in `model` with LoRALinear wrappers.

    Returns the number of wrapped Linear layers.
    """
    replaced = 0
    # Collect matching module paths first so we don't mutate while iterating
    to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(name.endswith(t) for t in targets):
            to_replace.append(name)

    for name in to_replace:
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        leaf_name = parts[-1]
        base_linear = getattr(parent, leaf_name)
        if isinstance(base_linear, LoRALinear):
            continue
        wrapped = LoRALinear(base_linear, r=r, alpha=alpha, dropout=dropout)
        wrapped.to(device=base_linear.weight.device, dtype=base_linear.weight.dtype)
        setattr(parent, leaf_name, wrapped)
        replaced += 1

    return replaced


def lora_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    """Yield only LoRA trainable parameters (lora_A, lora_B)."""
    for name, param in model.named_parameters():
        if ("lora_A" in name or "lora_B" in name) and param.requires_grad:
            yield param


def trainable_and_buffer_state_dict(model: nn.Module) -> dict:
    """Return a compact state_dict containing all trainable parameters and all buffers.

    This avoids saving ~1.2GB of frozen ViT weights on every checkpoint while
    preserving BatchNorm running stats and all trained parameters (CNN, decoder,
    fusion, LoRA, or full ViT when finetuning).
    """
    trainable_keys = {name for name, p in model.named_parameters() if p.requires_grad}
    buffer_keys = {name for name, _ in model.named_buffers()}
    keep_keys = trainable_keys | buffer_keys
    full_sd = model.state_dict()
    return {k: v.detach().cpu() for k, v in full_sd.items() if k in keep_keys}
