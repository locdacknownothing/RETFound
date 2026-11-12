import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt
from huggingface_hub import hf_hub_download

import models_vit
from util.pos_embed import interpolate_pos_embed
from util.lora import apply_lora


class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) * 2 used in the CNN encoder and U-Net decoder."""

    def __init__(self, in_ch: int, out_ch: int, mid_ch: Optional[int] = None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNBranch(nn.Module):
    """Full-resolution 5-stage CNN encoder at strides [1, 2, 4, 8, 16].

    Provides crisp high-frequency vessel boundary and 1-2 px capillary features
    at strides 1 and 2, plus multi-scale features at strides 4, 8, and 16 for
    fusion with the ViT pyramid.
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


class FuseBlock(nn.Module):
    """Concatenate CNN & ViT features, project with 1x1 conv, refine with 3x3 Conv-BN-ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        target_hw = tensors[0].shape[-2:]
        aligned = [
            t
            if t.shape[-2:] == target_hw
            else F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False)
            for t in tensors
        ]
        return self.fuse(torch.cat(aligned, dim=1))


class UpBlock(nn.Module):
    """Bilinear upsample by 2x + skip concatenation + DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


def load_retfound_backbone(
    vit: nn.Module,
    model_name: str,
    finetune_ckpt: str,
) -> Tuple[List[str], List[str]]:
    """Load pre-trained RETFound weights into `vit` with full key hygiene and validation.

    Follows main_finetune.py L183-L223 and raises RuntimeError if download fails
    or if any Transformer block weight (`blocks.*`) is missing after loading.
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

    # Key hygiene (matching main_finetune.py L208-L210)
    checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("mlp.w12.", "mlp.fc1."): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("mlp.w3.", "mlp.fc2."): v for k, v in checkpoint_model.items()}

    # Drop classification head keys if present
    for k in ["head.weight", "head.bias", "fc_norm.weight", "fc_norm.bias"]:
        if k in checkpoint_model and (
            k not in vit.state_dict() or checkpoint_model[k].shape != vit.state_dict()[k].shape
        ):
            del checkpoint_model[k]

    # Bicubic 2D interpolation of pos_embed to vit's default grid size (e.g. 448x448)
    interpolate_pos_embed(vit, checkpoint_model)

    msg = vit.load_state_dict(checkpoint_model, strict=False)
    missing_blocks = [k for k in msg.missing_keys if k.startswith("blocks.")]
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
    """Extracts 4-level spatial feature maps from RETFound ViT-L intermediate blocks."""

    def __init__(
        self,
        model_name: str = "RETFound_mae",
        img_size: int = 448,
        drop_path_rate: float = 0.1,
        vit_layers: Sequence[int] = (5, 11, 17, 23),
        out_channels: Tuple[int, int, int, int] = (128, 256, 256, 256),
        use_grad_checkpointing: bool = False,
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

        # Per-hook token LayerNorms before spatial reshaping
        self.token_norms = nn.ModuleList(
            [nn.LayerNorm(embed_dim, eps=1e-6) for _ in range(4)]
        )

        c_s4, c_s8, c_s16a, c_s16b = out_channels
        # Block 6 -> stride 4 (upsample x4 from patch grid)
        self.proj_s4 = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 192, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(192),
            nn.GELU(),
            nn.ConvTranspose2d(192, c_s4, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(c_s4),
            nn.GELU(),
        )
        # Block 12 -> stride 8 (upsample x2 from patch grid)
        self.proj_s8 = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, c_s8, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(c_s8),
            nn.GELU(),
        )
        # Block 18 -> stride 16
        self.proj_s16a = nn.Sequential(
            nn.Conv2d(embed_dim, c_s16a, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_s16a),
            nn.GELU(),
        )
        # Block 24 -> stride 16
        self.proj_s16b = nn.Sequential(
            nn.Conv2d(embed_dim, c_s16b, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_s16b),
            nn.GELU(),
        )

    def forward_tokens(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], Tuple[int, int]]:
        """Run ViT forward pass and collect intermediate spatial token grids."""
        B, C, H, W = x.shape
        ps = self.patch_size
        if H % ps != 0 or W % ps != 0:
            new_h = int(round(H / ps) * ps)
            new_w = int(round(W / ps) * ps)
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        else:
            new_h, new_w = H, W

        grid_h, grid_w = new_h // ps, new_w // ps

        tokens = self.vit.patch_embed(x)
        tokens = self.vit._pos_embed(tokens)
        if hasattr(self.vit, "patch_drop"):
            tokens = self.vit.patch_drop(tokens)
        if hasattr(self.vit, "norm_pre"):
            tokens = self.vit.norm_pre(tokens)

        collected: List[torch.Tensor] = []
        hook_map = {blk_idx: slot for slot, blk_idx in enumerate(self.vit_layers)}

        for idx, blk in enumerate(self.vit.blocks):
            if self.use_grad_checkpointing and self.training:
                tokens = grad_ckpt(blk, tokens, use_reentrant=False)
            else:
                tokens = blk(tokens)

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

        return collected, (grid_h, grid_w)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feats, _ = self.forward_tokens(x)
        p4 = self.proj_s4(feats[0])
        p8 = self.proj_s8(feats[1])
        p16a = self.proj_s16a(feats[2])
        p16b = self.proj_s16b(feats[3])
        return p4, p8, p16a, p16b


class RETFoundSegNet(nn.Module):
    """Hybrid CNN + RETFound ViT U-Net for retinal vessel segmentation."""

    def __init__(
        self,
        model_name: str = "RETFound_mae",
        in_chans: int = 3,
        num_classes: int = 1,
        base_ch: int = 32,
        img_size: int = 448,
        drop_path_rate: float = 0.1,
        vit_layers: Sequence[int] = (5, 11, 17, 23),
        adaptation: str = "lora",
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        deep_supervision: bool = True,
        grad_checkpointing: bool = False,
        finetune_ckpt: str = "",
    ):
        super().__init__()
        self.model_name = model_name
        self.adaptation = adaptation
        self.deep_supervision = bool(deep_supervision)
        self.use_vit = model_name in ("RETFound_mae", "RETFound_dinov2")

        self.cnn = CNNBranch(in_chans=in_chans, base_ch=base_ch)
        c1, c2, c4, c8, c16 = self.cnn.channels  # (32, 64, 128, 256, 512)

        if self.use_vit:
            eff_dp = 0.0 if adaptation == "frozen" else drop_path_rate
            use_gc = bool(grad_checkpointing) or (adaptation == "finetune")
            self.vit_pyramid = ViTPyramid(
                model_name=model_name,
                img_size=img_size,
                drop_path_rate=eff_dp,
                vit_layers=vit_layers,
                out_channels=(c4, c8, c8, c8),
                use_grad_checkpointing=use_gc,
            )
            if finetune_ckpt:
                load_retfound_backbone(self.vit_pyramid.vit, model_name, finetune_ckpt)

            if adaptation == "frozen":
                for p in self.vit_pyramid.vit.parameters():
                    p.requires_grad_(False)
                self.vit_pyramid.vit.eval()
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
                print(
                    f"[Adaptation] Applied LoRA (rank={lora_rank}, alpha={lora_alpha}) "
                    f"to {n_lora} ViT attention linear layers."
                )
            elif adaptation == "finetune":
                for p in self.vit_pyramid.vit.parameters():
                    p.requires_grad_(True)
            else:
                raise ValueError(f"Unknown adaptation mode: {adaptation}")

            self.fuse4 = FuseBlock(c4 + c4, c4)
            self.fuse8 = FuseBlock(c8 + c8, c8)
            self.fuse16 = FuseBlock(c16 + c8 + c8, c16)
        else:
            self.vit_pyramid = None
            self.fuse4 = None
            self.fuse8 = None
            self.fuse16 = None

        # U-Net Decoder
        self.up8 = UpBlock(c16, c8, c8)   # stride 16 -> 8
        self.up4 = UpBlock(c8, c4, c4)    # stride 8 -> 4
        self.up2 = UpBlock(c4, c2, c2)    # stride 4 -> 2
        self.up1 = UpBlock(c2, c1, c1)    # stride 2 -> 1

        self.head_s1 = nn.Conv2d(c1, num_classes, kernel_size=1)
        if self.deep_supervision:
            self.aux_s2 = nn.Conv2d(c2, num_classes, kernel_size=1)
            self.aux_s4 = nn.Conv2d(c4, num_classes, kernel_size=1)
            self.aux_s8 = nn.Conv2d(c8, num_classes, kernel_size=1)
        else:
            self.aux_s2 = self.aux_s4 = self.aux_s8 = None

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen ViT in eval mode so stochastic depth (drop_path) stays off
        if self.use_vit and self.adaptation == "frozen" and self.vit_pyramid is not None:
            self.vit_pyramid.vit.eval()
        return self

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Dict[str, Union[torch.Tensor, List[torch.Tensor]]]]:
        H, W = x.shape[-2:]
        c1, c2, c4, c8, c16 = self.cnn(x)

        if self.use_vit and self.vit_pyramid is not None:
            p4, p8, p16a, p16b = self.vit_pyramid(x)
            f4 = self.fuse4(c4, p4)
            f8 = self.fuse8(c8, p8)
            f16 = self.fuse16(c16, p16a, p16b)
        else:
            f4, f8, f16 = c4, c8, c16

        d8 = self.up8(f16, f8)
        d4 = self.up4(d8, f4)
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


def build_seg_model(args) -> RETFoundSegNet:
    vit_layers = [int(x) for x in str(args.vit_layers).split(",") if x.strip()]
    return RETFoundSegNet(
        model_name=args.model,
        in_chans=3,
        num_classes=1,
        base_ch=getattr(args, "base_ch", 32),
        img_size=args.crop_size,
        drop_path_rate=args.drop_path,
        vit_layers=vit_layers,
        adaptation=args.adaptation,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        deep_supervision=args.deep_supervision,
        grad_checkpointing=getattr(args, "grad_checkpointing", False),
        finetune_ckpt=args.finetune if args.model != "unet" else "",
    )
