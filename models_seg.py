import os
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple, Union

import timm
from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block as TimmViTBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt
from huggingface_hub import hf_hub_download

import models_vit
from util.pos_embed import interpolate_pos_embed
from util.lora import apply_lora


class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) * 2 used in the U-Net encoder and upsampling decoder."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        mid_ch: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=float(dropout)))
        layers.extend(
            [
                nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNBranch(nn.Module):
    """Full-resolution 5-stage CNN encoder at strides [1, 2, 4, 8, 16].

    Used strictly for the standalone `model == "unet"` baseline.
    """

    def __init__(
        self,
        in_chans: int = 3,
        base_ch: int = 32,
    ):
        super().__init__()
        c1, c2, c4, c8, c16 = (
            base_ch,
            base_ch * 2,
            base_ch * 4,
            base_ch * 8,
            base_ch * 16,
        )
        self.channels = (c1, c2, c4, c8, c16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stage1 = DoubleConv(in_chans, c1)
        self.stage2 = DoubleConv(c1, c2)
        self.stage4 = DoubleConv(c2, c4)
        self.stage8 = DoubleConv(c4, c8)
        self.stage16 = DoubleConv(c8, c16)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c1 = self.stage1(x)                # (B, 32, H, W)
        c2 = self.stage2(self.pool(c1))    # (B, 64, H/2, W/2)
        c4 = self.stage4(self.pool(c2))    # (B, 128, H/4, W/4)
        c8 = self.stage8(self.pool(c4))    # (B, 256, H/8, W/8)
        c16 = self.stage16(self.pool(c8))  # (B, 512, H/16, W/16)
        return c1, c2, c4, c8, c16


class UpBlock(nn.Module):
    """Bilinear upsample to skip resolution + skip concatenation + DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class PixelShuffleUpBlock(nn.Module):
    """Learned sub-pixel 2x upsampling + skip concatenation + DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.conv = DoubleConv(out_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ChannelGate(nn.Module):
    """CBAM channel attention using adaptive global avg/max pooling."""

    def __init__(self, gate_channels: int, reduction_ratio: int = 4):
        super().__init__()
        hidden_ch = max(4, gate_channels // reduction_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(gate_channels, hidden_ch),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_ch, gate_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = F.adaptive_avg_pool2d(x, 1).flatten(1)
        max_pool = F.adaptive_max_pool2d(x, 1).flatten(1)
        att = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        return x * att.unsqueeze(-1).unsqueeze(-1)


class SpatialGate(nn.Module):
    """CBAM spatial attention with 7x7 conv over channel-pooled maps."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_out = x.amax(dim=1, keepdim=True)
        mean_out = x.mean(dim=1, keepdim=True)
        scale = torch.sigmoid(self.spatial(torch.cat([max_out, mean_out], dim=1)))
        return x * scale


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Channel + Spatial attention)."""

    def __init__(self, gate_channels: int, reduction_ratio: int = 4):
        super().__init__()
        self.channel_gate = ChannelGate(gate_channels, reduction_ratio=reduction_ratio)
        self.spatial_gate = SpatialGate(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial_gate(self.channel_gate(x))


class PreAdapter(nn.Module):
    """Lightweight zero-initialized residual input conv before ViT patch embedding."""

    def __init__(self, in_chans: int = 3, mid_ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, mid_ch),
            nn.GELU(),
        )
        self.proj = nn.Conv2d(mid_ch, in_chans, kernel_size=1, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj(self.net(x))


class PatchEmbedConvStem(nn.Module):
    """Zero-initialized overlapping conv stem added inside ViTPyramid tokenizer.

    Encodes continuous sub-patch 1-2 px capillary cues across patch boundaries
    before `vit._pos_embed` while remaining an exact zero residual at step 0.
    """

    def __init__(self, in_chans: int = 3, embed_dim: int = 1024):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )
        self.proj = nn.Conv2d(128, embed_dim, kernel_size=1, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, grid_hw: Tuple[int, int]) -> torch.Tensor:
        gh, gw = int(grid_hw[0]), int(grid_hw[1])
        stem_2d = self.proj(self.stem(x))
        if stem_2d.shape[-2:] != (gh, gw):
            stem_2d = F.interpolate(
                stem_2d, size=(gh, gw), mode="bilinear", align_corners=False
            )
        return stem_2d


class SpaceAdapter(nn.Module):
    """Post-attention bottleneck adapter with multi-scale 2D DWConv spatial mixing.

    Explicitly separates prefix tokens (e.g. CLS token) and reshapes patch tokens using
    the explicit `(grid_h, grid_w)` tuple (never assuming square grids). Zero-initialized
    output projection guarantees exact identity at step 0.
    """

    def __init__(self, dim: int, hidden_dim: int = 48, num_prefix_tokens: int = 1):
        super().__init__()
        self.num_prefix_tokens = int(num_prefix_tokens)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        num_groups = 8 if hidden_dim % 8 == 0 else (4 if hidden_dim % 4 == 0 else 1)
        self.norm = nn.GroupNorm(num_groups, hidden_dim)
        self.dwconv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=True
        )
        self.dwconv_d2 = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=hidden_dim,
            bias=True,
        )
        self.fc2 = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor, grid_hw: Tuple[int, int]) -> torch.Tensor:
        gh, gw = int(grid_hw[0]), int(grid_hw[1])
        h = self.act(self.fc1(x))

        if self.num_prefix_tokens > 0:
            prefix = h[:, : self.num_prefix_tokens, :]
            patches = h[:, self.num_prefix_tokens :, :]
        else:
            prefix = None
            patches = h

        B, N, r = patches.shape
        if N != gh * gw:
            raise ValueError(
                f"Patch token count N={N} does not match explicit grid_hw=({gh}, {gw})"
            )

        patches_2d = patches.transpose(1, 2).contiguous().reshape(B, r, gh, gw)
        ms = self.dwconv(patches_2d) + self.dwconv_d2(patches_2d)
        patches_2d = patches_2d + self.act(self.norm(ms))
        patches = patches_2d.reshape(B, r, N).transpose(1, 2).contiguous()

        if prefix is not None:
            h = torch.cat([prefix, patches], dim=1)
        else:
            h = patches

        return x + self.fc2(h)


class MLPAdapter(nn.Module):
    """Parallel bottleneck MLP adapter with zero-initialized output projection."""

    def __init__(self, dim: int, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class AdaptedViTBlock(nn.Module):
    """Wraps a timm ViT Block in-place while preserving exact `blocks.{i}.*` state_dict keys.

    Integrates:
    - Post-attention `space_adapter` (with multi-scale 2D DWConv on reshaped patch tokens)
    - Parallel `0.5`-scaled `mlp_adapter` alongside frozen `mlp`
    """

    def __init__(
        self,
        base_block: nn.Module,
        dim: int = 1024,
        space_adapter_dim: int = 48,
        mlp_adapter_dim: int = 32,
        mlp_scale: float = 0.5,
        num_prefix_tokens: int = 1,
    ):
        super().__init__()
        self.norm1 = base_block.norm1
        self.attn = base_block.attn
        self.ls1 = base_block.ls1
        self.drop_path1 = base_block.drop_path1
        self.norm2 = base_block.norm2
        self.mlp = base_block.mlp
        self.ls2 = base_block.ls2
        self.drop_path2 = base_block.drop_path2

        self.mlp_scale = float(mlp_scale)
        self.space_adapter = SpaceAdapter(
            dim=dim, hidden_dim=space_adapter_dim, num_prefix_tokens=num_prefix_tokens
        )
        self.mlp_adapter = MLPAdapter(dim=dim, hidden_dim=mlp_adapter_dim)

    def forward(self, x: torch.Tensor, grid_hw: Tuple[int, int]) -> torch.Tensor:
        attn_out = self.space_adapter(self.attn(self.norm1(x)), grid_hw)
        x = x + self.drop_path1(self.ls1(attn_out))
        xn = self.norm2(x)
        x = x + self.drop_path2(self.ls2(self.mlp(xn))) + self.mlp_scale * self.mlp_adapter(xn)
        return x


class SpatialMaskNorm(nn.Module):
    """Normalize class-similarity logits across spatial patch tokens per class."""

    def __init__(self, num_classes: int = 1, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


def build_2d_sincos_pos_embed(
    gh: int,
    gw: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    temperature: float = 20.0,
) -> torch.Tensor:
    """Build parameter-free 2D sin-cos positional embeddings `(1, gh*gw, dim)` for arbitrary grids."""
    if dim % 4 != 0:
        raise ValueError(f"2D sincos pos embed requires dim % 4 == 0, got {dim}")
    half_dim = dim // 2
    omega = torch.arange(half_dim // 2, device=device, dtype=torch.float32)
    omega = 1.0 / (float(temperature) ** (omega / (half_dim / 2.0)))

    y = torch.arange(gh, device=device, dtype=torch.float32)
    x = torch.arange(gw, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

    out_y = grid_y.reshape(-1, 1) * omega.reshape(1, -1)
    out_x = grid_x.reshape(-1, 1) * omega.reshape(1, -1)
    pos = torch.cat([out_y.sin(), out_y.cos(), out_x.sin(), out_x.cos()], dim=1)
    return pos.unsqueeze(0).to(dtype=dtype)


class MaskTransformerDecoder(nn.Module):
    """Sequential 2-layer Transformer Decoder placed after ViT `norm`.

    Refines global patch context with learnable vessel query tokens (`cls_emb`)
    and concatenates the decoded spatial features with the L2-normalized
    class-similarity maps (`(B, d_dec + num_queries, grid_h, grid_w) -> (B, d_dec, grid_h, grid_w)`),
    avoiding FunduSegmenter's 1-channel bottleneck collapse.
    """

    def __init__(
        self,
        d_encoder: int = 1024,
        d_dec: int = 192,
        n_layers: int = 2,
        n_heads: int = 6,
        mlp_ratio: float = 2.0,
        num_queries: int = 4,
        drop_path_rate: float = 0.05,
    ):
        super().__init__()
        self.d_dec = int(d_dec)
        self.num_queries = int(num_queries)

        self.proj_dec = nn.Linear(d_encoder, d_dec)
        self.cls_emb = nn.Parameter(torch.zeros(1, num_queries, d_dec))
        trunc_normal_(self.cls_emb, std=0.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [
                TimmViTBlock(
                    dim=d_dec,
                    num_heads=n_heads,
                    mlp_ratio=float(mlp_ratio),
                    qkv_bias=True,
                    proj_drop=0.1,
                    attn_drop=0.0,
                    drop_path=dpr[i],
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                )
                for i in range(n_layers)
            ]
        )
        self.decoder_norm = nn.LayerNorm(d_dec, eps=1e-6)
        self.proj_patch = nn.Linear(d_dec, d_dec, bias=False)
        self.proj_classes = nn.Linear(d_dec, d_dec, bias=False)
        self.mask_norm = SpatialMaskNorm(num_classes=num_queries)
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(d_dec, d_dec, kernel_size=3, padding=1, groups=d_dec, bias=False),
            nn.BatchNorm2d(d_dec),
            nn.GELU(),
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(d_dec + num_queries, d_dec, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_dec),
            nn.GELU(),
        )

    def forward(
        self, patch_tokens: torch.Tensor, grid_hw: Tuple[int, int]
    ) -> torch.Tensor:
        gh, gw = int(grid_hw[0]), int(grid_hw[1])
        B, N, _ = patch_tokens.shape
        if N != gh * gw:
            raise ValueError(f"Expected N={gh * gw} tokens, got {N}")

        x = self.proj_dec(patch_tokens)
        x = x + build_2d_sincos_pos_embed(gh, gw, self.d_dec, x.device, x.dtype)
        cls_emb = self.cls_emb.expand(B, -1, -1)
        x = torch.cat([x, cls_emb], dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        patches = x[:, : -self.num_queries, :]
        cls_feat = x[:, -self.num_queries :, :]

        patches_norm = F.normalize(self.proj_patch(patches), dim=-1)
        cls_norm = F.normalize(self.proj_classes(cls_feat), dim=-1)
        cls_map = torch.bmm(patches_norm, cls_norm.transpose(1, 2))
        cls_map = self.mask_norm(cls_map)

        patches_2d = patches.transpose(1, 2).contiguous().reshape(B, self.d_dec, gh, gw)
        patches_2d = patches_2d + self.spatial_refine(patches_2d)
        cls_map_2d = cls_map.transpose(1, 2).contiguous().reshape(B, self.num_queries, gh, gw)
        return self.out_proj(torch.cat([patches_2d, cls_map_2d], dim=1))


class ProgressiveSkipProj(nn.Module):
    """Projects a ViT intermediate feature map through learned sub-pixel 2x stages + CBAM.

    Uses an initial 1x1 bottleneck (e.g. 1024 -> 96) followed by PixelShuffle(2) +
    3x3 Conv-BN-GELU at each stage to preserve sub-patch 1-2 px vessel phase
    without bilinear smearing or parameter bloat.
    """

    def __init__(
        self,
        in_ch: int,
        stage_channels: Sequence[int],
        bottleneck_ch: Optional[int] = 96,
    ):
        super().__init__()
        if len(stage_channels) < 1:
            raise ValueError("stage_channels must have at least 1 stage")

        if bottleneck_ch is not None and int(bottleneck_ch) > 0:
            b_ch = int(bottleneck_ch)
            self.in_proj = nn.Sequential(
                nn.Conv2d(int(in_ch), b_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(b_ch),
                nn.GELU(),
            )
            prev_ch = b_ch
        else:
            self.in_proj = nn.Identity()
            prev_ch = int(in_ch)

        stages = []
        for out_ch in stage_channels:
            out_ch = int(out_ch)
            stages.append(
                nn.Sequential(
                    nn.Conv2d(prev_ch, out_ch * 4, kernel_size=1, bias=False),
                    nn.PixelShuffle(2),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.GELU(),
                )
            )
            prev_ch = out_ch
        self.stages = nn.ModuleList(stages)
        self.cbam = CBAM(prev_ch, reduction_ratio=4)

    def forward(
        self, x: torch.Tensor, target_sizes: Sequence[Tuple[int, int]]
    ) -> torch.Tensor:
        if len(target_sizes) != len(self.stages):
            raise ValueError(
                f"Expected {len(self.stages)} target sizes, got {len(target_sizes)}"
            )
        x = self.in_proj(x)
        for stage, size_hw in zip(self.stages, target_sizes):
            x = stage(x)
            if x.shape[-2:] != tuple(size_hw):
                x = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
        return self.cbam(x)


def load_retfound_backbone(
    vit: nn.Module,
    model_name: str,
    finetune_ckpt: str,
) -> Tuple[List[str], List[str]]:
    """Load pre-trained RETFound weights into `vit` with full key hygiene and validation.

    Raises RuntimeError if download fails or if any non-adapter Transformer block
    weight (`blocks.*`) is missing after loading.
    """
    if not finetune_ckpt:
        raise ValueError("`finetune_ckpt` must be provided to load pre-trained RETFound weights.")

    if os.path.isfile(finetune_ckpt):
        checkpoint_path = finetune_ckpt
        print(f"[Backbone] Loading local checkpoint: {checkpoint_path}")
    else:
        print(f"[Backbone] Downloading from HuggingFace Hub: YukunZhou/{finetune_ckpt}")
        checkpoint_path = hf_hub_download(
            repo_id=f"YukunZhou/{finetune_ckpt}",
            filename=f"{finetune_ckpt}.pth",
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        if model_name == "RETFound_dinov2" and "teacher" in checkpoint:
            checkpoint_model = checkpoint["teacher"]
        elif "model" in checkpoint:
            checkpoint_model = checkpoint["model"]
        elif "state_dict" in checkpoint:
            checkpoint_model = checkpoint["state_dict"]
        else:
            checkpoint_model = checkpoint
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")

    checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("mlp.w12.", "mlp.fc1."): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("mlp.w3.", "mlp.fc2."): v for k, v in checkpoint_model.items()}

    for k in ["head.weight", "head.bias", "fc_norm.weight", "fc_norm.bias"]:
        if k in checkpoint_model and (
            k not in vit.state_dict() or checkpoint_model[k].shape != vit.state_dict()[k].shape
        ):
            del checkpoint_model[k]

    interpolate_pos_embed(vit, checkpoint_model)

    msg = vit.load_state_dict(checkpoint_model, strict=False)
    missing_blocks = [
        k for k in msg.missing_keys if k.startswith("blocks.") and "adapter" not in k.lower()
    ]
    if missing_blocks:
        raise RuntimeError(
            f"Critical error loading {model_name}: missing block keys: {missing_blocks[:10]}"
        )

    print(
        f"[Backbone] Successfully loaded {model_name} from {checkpoint_path} | "
        f"missing={msg.missing_keys}, unexpected={len(msg.unexpected_keys)}"
    )
    return list(msg.missing_keys), list(msg.unexpected_keys)


class ViTPyramid(nn.Module):
    """Adapted RETFound ViT-L backbone extracting 4-level and final spatial tokens."""

    def __init__(
        self,
        model_name: str = "RETFound_mae",
        img_size: int = 448,
        drop_path_rate: float = 0.1,
        vit_layers: Sequence[int] = (5, 11, 17, 23),
        space_adapter_dim: int = 48,
        mlp_adapter_dim: int = 32,
        adapter_mlp_scale: float = 0.5,
        use_grad_checkpointing: bool = False,
        finetune_ckpt: str = "",
    ):
        super().__init__()
        self.model_name = model_name
        self.vit_layers = tuple(int(i) for i in vit_layers)
        if len(self.vit_layers) != 4:
            raise ValueError(f"Expected 4 vit_layers, got {self.vit_layers}")
        self.use_grad_checkpointing = bool(use_grad_checkpointing)

        if model_name == "RETFound_mae":
            self.vit = models_vit.RETFound_mae(
                img_size=img_size,
                dynamic_img_size=True,
                num_classes=0,
                drop_path_rate=drop_path_rate,
                global_pool=False,
            )
            self.patch_size = 16
        elif model_name == "RETFound_dinov2":
            self.vit = timm.create_model(
                "vit_large_patch14_dinov2.lvd142m",
                pretrained=False,
                img_size=img_size,
                dynamic_img_size=True,
                num_classes=0,
                drop_path_rate=drop_path_rate,
            )
            self.patch_size = 14
        else:
            raise ValueError(
                f"Unsupported ViTPyramid model_name '{model_name}'. "
                f"Expected 'RETFound_mae' or 'RETFound_dinov2'."
            )

        embed_dim = int(getattr(self.vit, "embed_dim", 1024))
        self.embed_dim = embed_dim
        self.num_prefix_tokens = int(getattr(self.vit, "num_prefix_tokens", 1))

        # Load pre-trained RETFound weights BEFORE wrapping blocks with adapters
        if finetune_ckpt:
            load_retfound_backbone(self.vit, model_name, finetune_ckpt)

        # Zero-initialized overlapping conv stem inside ViT tokenizer
        self.patch_stem_adapter = PatchEmbedConvStem(in_chans=3, embed_dim=embed_dim)

        # Wrap every ViT block with Space_Adapter + parallel 0.5-scaled MLP_Adapter
        self.vit.blocks = nn.ModuleList(
            [
                AdaptedViTBlock(
                    blk,
                    dim=embed_dim,
                    space_adapter_dim=space_adapter_dim,
                    mlp_adapter_dim=mlp_adapter_dim,
                    mlp_scale=adapter_mlp_scale,
                    num_prefix_tokens=self.num_prefix_tokens,
                )
                for blk in self.vit.blocks
            ]
        )

        self.token_norms = nn.ModuleList(
            [nn.LayerNorm(embed_dim, eps=1e-6) for _ in range(4)]
        )

    def forward_tokens(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor, Tuple[int, int]]:
        """Run adapted ViT forward pass and return 4 intermediate 2D grids + final tokens."""
        B, _, H, W = x.shape
        ps = self.patch_size
        if H % ps != 0 or W % ps != 0:
            new_h = int(round(H / ps) * ps)
            new_w = int(round(W / ps) * ps)
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        else:
            new_h, new_w = H, W

        grid_h, grid_w = new_h // ps, new_w // ps
        grid_hw = (grid_h, grid_w)

        tokens = self.vit.patch_embed(x)
        stem_2d = self.patch_stem_adapter(x, grid_hw)
        if tokens.ndim == 4:
            if tokens.shape[1:3] == grid_hw:
                tokens = tokens + stem_2d.permute(0, 2, 3, 1)
            else:
                tokens = tokens + stem_2d
        else:
            tokens = tokens + stem_2d.flatten(2).transpose(1, 2).contiguous()
        tokens = self.vit._pos_embed(tokens)
        if hasattr(self.vit, "patch_drop"):
            tokens = self.vit.patch_drop(tokens)
        if hasattr(self.vit, "norm_pre"):
            tokens = self.vit.norm_pre(tokens)

        collected: List[torch.Tensor] = []
        hook_map = {blk_idx: slot for slot, blk_idx in enumerate(self.vit_layers)}

        for idx, blk in enumerate(self.vit.blocks):
            if self.use_grad_checkpointing and self.training:
                tokens = grad_ckpt(blk, tokens, grid_hw, use_reentrant=False)
            else:
                tokens = blk(tokens, grid_hw)

            if idx in hook_map:
                slot = hook_map[idx]
                normed = self.token_norms[slot](tokens)
                spatial = normed[:, self.num_prefix_tokens :, :]
                spatial_2d = (
                    spatial.transpose(1, 2)
                    .contiguous()
                    .reshape(B, self.embed_dim, grid_h, grid_w)
                )
                collected.append(spatial_2d)

        final_normed = self.vit.norm(tokens)
        final_patch_tokens = final_normed[:, self.num_prefix_tokens :, :]
        return collected, final_patch_tokens, grid_hw


class UNetSegNet(nn.Module):
    """Standalone 5-stage U-Net baseline (used strictly when `model == "unet"`)."""

    def __init__(
        self,
        model_name: str = "unet",
        in_chans: int = 3,
        num_classes: int = 1,
        base_ch: int = 32,
        deep_supervision: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.adaptation = "full"
        self.deep_supervision = bool(deep_supervision)
        self.use_vit = False

        self.cnn = CNNBranch(in_chans=in_chans, base_ch=base_ch)
        c1, c2, c4, c8, c16 = self.cnn.channels

        self.up8 = UpBlock(c16, c8, c8)
        self.up4 = UpBlock(c8, c4, c4)
        self.up2 = UpBlock(c4, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)

        self.head_s1 = nn.Conv2d(c1, num_classes, kernel_size=1)
        if self.deep_supervision:
            self.aux_s2 = nn.Conv2d(c2, num_classes, kernel_size=1)
            self.aux_s4 = nn.Conv2d(c4, num_classes, kernel_size=1)
            self.aux_s8 = nn.Conv2d(c8, num_classes, kernel_size=1)
        else:
            self.aux_s2 = self.aux_s4 = self.aux_s8 = None

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Dict[str, Union[torch.Tensor, List[torch.Tensor]]]]:
        H, W = x.shape[-2:]
        c1, c2, c4, c8, c16 = self.cnn(x)

        d8 = self.up8(c16, c8)
        d4 = self.up4(d8, c4)
        d2 = self.up2(d4, c2)
        d1 = self.up1(d2, c1)

        logits_s1 = self.head_s1(d1)
        if self.training and self.deep_supervision and self.aux_s2 is not None:
            logits_s2 = F.interpolate(
                self.aux_s2(d2), size=(H, W), mode="bilinear", align_corners=False
            )
            logits_s4 = F.interpolate(
                self.aux_s4(d4), size=(H, W), mode="bilinear", align_corners=False
            )
            logits_s8 = F.interpolate(
                self.aux_s8(d8), size=(H, W), mode="bilinear", align_corners=False
            )
            return {
                "logits": logits_s1,
                "aux_logits": [logits_s2, logits_s4, logits_s8],
            }
        return logits_s1


class RETFoundSegNet(nn.Module):
    """Unified RETFound segmentation architecture without parallel CNN encoder branch.

    Pipeline:
      1. Zero-initialized residual input `PreAdapter` (`3 -> 32 -> 32 -> 3`)
      2. Adapted RETFound ViT-L/16 encoder (`AdaptedViTBlock` in all 24 blocks)
      3. Sequential `MaskTransformerDecoder` after `vit.norm` (`d_dec=256` + query similarity map)
      4. Direct 4-level ViT feature projections (`[blk23->s8, blk17->s4, blk11->s2, blk5->s1]`)
         concatenated directly to progressive CNN upsampling layers (`up8 -> up4 -> up2 -> up1`).
    """

    def __init__(
        self,
        model_name: str = "RETFound_mae",
        in_chans: int = 3,
        num_classes: int = 1,
        img_size: int = 592,
        drop_path_rate: float = 0.1,
        vit_layers: Sequence[int] = (5, 11, 17, 23),
        adaptation: str = "frozen",
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        space_adapter_dim: int = 48,
        mlp_adapter_dim: int = 32,
        adapter_mlp_scale: float = 0.5,
        d_dec: int = 192,
        unfreeze_patch_embed: bool = True,
        deep_supervision: bool = True,
        grad_checkpointing: bool = False,
        finetune_ckpt: str = "",
    ):
        super().__init__()
        self.model_name = model_name
        self.adaptation = adaptation
        self.unfreeze_patch_embed = bool(unfreeze_patch_embed)
        self.deep_supervision = bool(deep_supervision)
        self.use_vit = True

        self.pre_adapter = PreAdapter(in_chans=in_chans, mid_ch=32)

        eff_dp = 0.0 if adaptation == "frozen" else drop_path_rate
        use_gc = bool(grad_checkpointing) or (adaptation == "finetune")
        self.vit_pyramid = ViTPyramid(
            model_name=model_name,
            img_size=img_size,
            drop_path_rate=eff_dp,
            vit_layers=vit_layers,
            space_adapter_dim=space_adapter_dim,
            mlp_adapter_dim=mlp_adapter_dim,
            adapter_mlp_scale=adapter_mlp_scale,
            use_grad_checkpointing=use_gc,
            finetune_ckpt=finetune_ckpt,
        )

        if adaptation == "frozen":
            for name, p in self.vit_pyramid.vit.named_parameters():
                if "adapter" in name.lower():
                    p.requires_grad_(True)
                else:
                    p.requires_grad_(False)
            if self.unfreeze_patch_embed:
                self.vit_pyramid.vit.patch_embed.requires_grad_(True)
                if hasattr(self.vit_pyramid.vit, "norm") and self.vit_pyramid.vit.norm is not None:
                    self.vit_pyramid.vit.norm.requires_grad_(True)
            self.vit_pyramid.vit.eval()
            for blk in self.vit_pyramid.vit.blocks:
                blk.space_adapter.train(True)
                blk.mlp_adapter.train(True)
            if self.unfreeze_patch_embed:
                self.vit_pyramid.vit.patch_embed.train(True)
                if hasattr(self.vit_pyramid.vit, "norm") and self.vit_pyramid.vit.norm is not None:
                    self.vit_pyramid.vit.norm.train(True)
        elif adaptation == "lora":
            for p in self.vit_pyramid.vit.parameters():
                p.requires_grad_(False)
            n_lora = apply_lora(
                self.vit_pyramid.vit,
                r=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                targets=("attn.qkv", "attn.proj"),
            )
            for name, p in self.vit_pyramid.vit.named_parameters():
                if "adapter" in name.lower():
                    p.requires_grad_(True)
            if self.unfreeze_patch_embed:
                self.vit_pyramid.vit.patch_embed.requires_grad_(True)
                if hasattr(self.vit_pyramid.vit, "norm") and self.vit_pyramid.vit.norm is not None:
                    self.vit_pyramid.vit.norm.requires_grad_(True)
            print(
                f"[Adaptation] Applied LoRA (rank={lora_rank}, alpha={lora_alpha}) "
                f"to {n_lora} ViT attention linear layers + per-block adapters"
                f"{' + unfrozen patch_embed/norm' if self.unfreeze_patch_embed else ''}."
            )
        elif adaptation == "finetune":
            for p in self.vit_pyramid.vit.parameters():
                p.requires_grad_(True)
        else:
            raise ValueError(f"Unknown adaptation mode: {adaptation}")

        embed_dim = self.vit_pyramid.embed_dim
        self.transformer_decoder = MaskTransformerDecoder(
            d_encoder=embed_dim,
            d_dec=d_dec,
            n_layers=2,
            n_heads=6,
            mlp_ratio=2.0,
            num_queries=4,
            drop_path_rate=0.05,
        )

        # Direct 4-level ViT feature projections -> strides [8, 4, 2, 1]
        # Bottleneck=256 preserves all 16x16=256 sub-patch spatial degrees of freedom
        c8, c4, c2, c1 = 128, 96, 64, 48
        skip_bottleneck = 256
        self.skip_s8 = ProgressiveSkipProj(embed_dim, [c8], bottleneck_ch=skip_bottleneck)
        self.skip_s4 = ProgressiveSkipProj(embed_dim, [c8, c4], bottleneck_ch=skip_bottleneck)
        self.skip_s2 = ProgressiveSkipProj(embed_dim, [c8, c4, c2], bottleneck_ch=skip_bottleneck)
        self.skip_s1 = ProgressiveSkipProj(embed_dim, [c8, c4, c2, c1], bottleneck_ch=skip_bottleneck)

        # Progressive sub-pixel upsampling CNN decoder with mild spatial dropout on coarse stages
        self.up8 = PixelShuffleUpBlock(d_dec, c8, c8, dropout=0.05)
        self.up4 = PixelShuffleUpBlock(c8, c4, c4, dropout=0.05)
        self.up2 = PixelShuffleUpBlock(c4, c2, c2, dropout=0.0)
        self.up1 = PixelShuffleUpBlock(c2, c1, c1, dropout=0.0)

        self.head_s1 = nn.Conv2d(c1, num_classes, kernel_size=1)
        if self.deep_supervision:
            self.aux_s2 = nn.Conv2d(c2, num_classes, kernel_size=1)
            self.aux_s4 = nn.Conv2d(c4, num_classes, kernel_size=1)
            self.aux_s8 = nn.Conv2d(c8, num_classes, kernel_size=1)
        else:
            self.aux_s2 = self.aux_s4 = self.aux_s8 = None

    def train(self, mode: bool = True):
        super().train(mode)
        if self.adaptation == "frozen":
            # Keep frozen ViT backbone in eval (disables drop_path), while adapters & unfrozen tokenizer follow `mode`
            self.vit_pyramid.vit.eval()
            for blk in self.vit_pyramid.vit.blocks:
                if hasattr(blk, "space_adapter"):
                    blk.space_adapter.train(mode)
                if hasattr(blk, "mlp_adapter"):
                    blk.mlp_adapter.train(mode)
            if self.unfreeze_patch_embed:
                self.vit_pyramid.vit.patch_embed.train(mode)
                if hasattr(self.vit_pyramid.vit, "norm") and self.vit_pyramid.vit.norm is not None:
                    self.vit_pyramid.vit.norm.train(mode)
        return self

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Dict[str, Union[torch.Tensor, List[torch.Tensor]]]]:
        H, W = x.shape[-2:]
        x = self.pre_adapter(x)
        feats, final_patch_tokens, grid_hw = self.vit_pyramid.forward_tokens(x)

        dec_s16 = self.transformer_decoder(final_patch_tokens, grid_hw)

        target_sizes = [
            (max(1, H // 8), max(1, W // 8)),
            (max(1, H // 4), max(1, W // 4)),
            (max(1, H // 2), max(1, W // 2)),
            (H, W),
        ]
        s8 = self.skip_s8(feats[3], target_sizes[:1])
        s4 = self.skip_s4(feats[2], target_sizes[:2])
        s2 = self.skip_s2(feats[1], target_sizes[:3])
        s1 = self.skip_s1(feats[0], target_sizes[:4])

        d8 = self.up8(dec_s16, s8)
        d4 = self.up4(d8, s4)
        d2 = self.up2(d4, s2)
        d1 = self.up1(d2, s1)

        logits_s1 = self.head_s1(d1)
        if logits_s1.shape[-2:] != (H, W):
            logits_s1 = F.interpolate(
                logits_s1, size=(H, W), mode="bilinear", align_corners=False
            )

        if self.training and self.deep_supervision and self.aux_s2 is not None:
            logits_s2 = F.interpolate(
                self.aux_s2(d2), size=(H, W), mode="bilinear", align_corners=False
            )
            logits_s4 = F.interpolate(
                self.aux_s4(d4), size=(H, W), mode="bilinear", align_corners=False
            )
            logits_s8 = F.interpolate(
                self.aux_s8(d8), size=(H, W), mode="bilinear", align_corners=False
            )
            return {
                "logits": logits_s1,
                "aux_logits": [logits_s2, logits_s4, logits_s8],
            }

        return logits_s1


def build_seg_model(args) -> Union[UNetSegNet, RETFoundSegNet]:
    if args.model == "unet":
        return UNetSegNet(
            model_name=args.model,
            in_chans=3,
            num_classes=1,
            base_ch=getattr(args, "base_ch", 32),
            deep_supervision=args.deep_supervision,
        )

    vit_layers = [int(x) for x in str(args.vit_layers).split(",") if x.strip()]
    return RETFoundSegNet(
        model_name=args.model,
        in_chans=3,
        num_classes=1,
        img_size=args.crop_size,
        drop_path_rate=args.drop_path,
        vit_layers=vit_layers,
        adaptation=args.adaptation,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        deep_supervision=args.deep_supervision,
        grad_checkpointing=getattr(args, "grad_checkpointing", False),
        finetune_ckpt=args.finetune,
    )
