from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Differentiable Soft-Dice loss over probabilities."""

    def __init__(self, smooth: float = 1.0, batch_dice: bool = True):
        super().__init__()
        self.smooth = float(smooth)
        self.batch_dice = bool(batch_dice)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        if self.batch_dice:
            dims = (0, 1, 2, 3) if probs.ndim == 4 else (0, 1, 2)
        else:
            dims = (1, 2, 3) if probs.ndim == 4 else (1, 2)

        intersection = (probs * targets).sum(dim=dims)
        denom = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return (1.0 - dice).mean()


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skel(img: torch.Tensor, iter_: int = 10) -> torch.Tensor:
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class SoftClDiceLoss(nn.Module):
    """Differentiable centerline-Dice (soft-clDice) loss for topology preservation."""

    def __init__(self, iter_: int = 10, smooth: float = 1.0):
        super().__init__()
        self.iter = int(iter_)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type=logits.device.type, enabled=False):
            y_pred = torch.sigmoid(logits.float())
            y_true = targets.float()
            skel_pred = soft_skel(y_pred, self.iter)
            skel_true = soft_skel(y_true, self.iter)
            tprec = ((skel_pred * y_true).sum(dim=(1, 2, 3)) + self.smooth) / (
                skel_pred.sum(dim=(1, 2, 3)) + self.smooth
            )
            tsens = ((skel_true * y_pred).sum(dim=(1, 2, 3)) + self.smooth) / (
                skel_true.sum(dim=(1, 2, 3)) + self.smooth
            )
            cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens + 1e-8)
            return cl_dice.mean()


class SkeletonRecallLoss(nn.Module):
    """Soft recall penalty weighted on dilated ground-truth vessel skeletons."""

    def __init__(self, iter_: int = 10, smooth: float = 1.0):
        super().__init__()
        self.iter = int(iter_)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type=logits.device.type, enabled=False):
            probs = torch.sigmoid(logits.float())
            targets_f = targets.float()
            with torch.no_grad():
                skel_gt = soft_dilate((soft_skel(targets_f, self.iter) > 0.1).float())
            intersection = (probs * skel_gt).sum(dim=(1, 2, 3))
            denom = skel_gt.sum(dim=(1, 2, 3))
            recall = (intersection + self.smooth) / (denom + self.smooth)
            return (1.0 - recall).mean()


class VesselSegmentationLoss(nn.Module):
    """Composite segmentation loss with deep supervision and topology warmup support.

    Supports additive combination of `bce`, `dice`, `cldice`, and `skelrec`.
    A term is active iff included in `loss_terms` AND its weight > 0.
    Topology losses (`cldice`, `skelrec`) run in fp32 on the main head only and
    are scaled by `topo_scale` for linear warmup ramping.
    """

    VALID_TERMS = frozenset({"bce", "dice", "cldice", "skelrec"})

    def __init__(
        self,
        loss_terms: str = "bce,dice,cldice,skelrec",
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        cldice_weight: float = 0.3,
        skelrec_weight: float = 0.2,
        cldice_iter: int = 10,
        pos_weight: Optional[float] = None,
        aux_weights: Tuple[float, ...] = (0.5, 0.25, 0.125),
    ):
        super().__init__()
        terms = [t.strip().lower() for t in loss_terms.split(",") if t.strip()]
        unknown = [t for t in terms if t not in self.VALID_TERMS]
        if unknown:
            raise ValueError(
                f"Unknown loss term(s) {unknown} in '{loss_terms}'. "
                f"Valid terms: {sorted(self.VALID_TERMS)}"
            )

        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.cldice_weight = float(cldice_weight)
        self.skelrec_weight = float(skelrec_weight)
        self.cldice_iter = int(cldice_iter)
        self.aux_weights = tuple(float(w) for w in aux_weights)

        self.use_bce = ("bce" in terms) and (self.bce_weight > 0.0)
        self.use_dice = ("dice" in terms) and (self.dice_weight > 0.0)
        self.use_cldice = ("cldice" in terms) and (self.cldice_weight > 0.0)
        self.use_skelrec = ("skelrec" in terms) and (self.skelrec_weight > 0.0)

        if not (self.use_bce or self.use_dice or self.use_cldice or self.use_skelrec):
            raise ValueError(
                f"No active loss terms resolved from loss_terms='{loss_terms}' "
                f"with weights (bce={self.bce_weight}, dice={self.dice_weight}, "
                f"cldice={self.cldice_weight}, skelrec={self.skelrec_weight})."
            )

        if pos_weight is not None and pos_weight > 0:
            self.register_buffer("pos_weight_buf", torch.tensor([float(pos_weight)]))
        else:
            self.pos_weight_buf = None

        self.dice = SoftDiceLoss(smooth=1.0, batch_dice=True) if self.use_dice else None
        self.cldice = (
            SoftClDiceLoss(iter_=self.cldice_iter, smooth=1.0)
            if self.use_cldice
            else None
        )
        self.skelrec = (
            SkeletonRecallLoss(iter_=self.cldice_iter, smooth=1.0)
            if self.use_skelrec
            else None
        )

    def _single_scale_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        include_topo: bool = True,
        topo_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(
                logits, size=targets.shape[-2:], mode="bilinear", align_corners=False
            )

        targets = targets.float()
        total = logits.new_zeros(())
        breakdown: Dict[str, float] = {}

        if self.use_bce:
            pw = (
                self.pos_weight_buf.to(device=logits.device, dtype=logits.dtype)
                if self.pos_weight_buf is not None
                else None
            )
            l_bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
            total = total + self.bce_weight * l_bce
            breakdown["bce"] = float(l_bce.detach().item())

        if self.use_dice and self.dice is not None:
            l_dice = self.dice(logits, targets)
            total = total + self.dice_weight * l_dice
            breakdown["dice"] = float(l_dice.detach().item())

        if include_topo:
            eff_scale = float(max(0.0, topo_scale))
            if self.use_cldice and self.cldice is not None:
                if eff_scale > 0.0:
                    l_cld = self.cldice(logits, targets)
                    total = total + (self.cldice_weight * eff_scale) * l_cld
                    breakdown["cldice"] = float(l_cld.detach().item())
                else:
                    with torch.no_grad():
                        l_cld = self.cldice(logits.detach(), targets)
                    breakdown["cldice"] = float(l_cld.item())

            if self.use_skelrec and self.skelrec is not None:
                if eff_scale > 0.0:
                    l_sk = self.skelrec(logits, targets)
                    total = total + (self.skelrec_weight * eff_scale) * l_sk
                    breakdown["skelrec"] = float(l_sk.detach().item())
                else:
                    with torch.no_grad():
                        l_sk = self.skelrec(logits.detach(), targets)
                    breakdown["skelrec"] = float(l_sk.item())

            if self.use_cldice or self.use_skelrec:
                breakdown["topo_scale"] = eff_scale

        return total, breakdown

    def forward(
        self,
        outputs: Union[torch.Tensor, Dict[str, Union[torch.Tensor, List[torch.Tensor]]]],
        targets: torch.Tensor,
        topo_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if isinstance(outputs, dict):
            main_logits = outputs["logits"]
            aux_list = outputs.get("aux_logits", None)
        elif isinstance(outputs, (tuple, list)):
            main_logits, aux_list = outputs[0], outputs[1]
        else:
            main_logits, aux_list = outputs, None

        main_loss, stats = self._single_scale_loss(
            main_logits, targets, include_topo=True, topo_scale=topo_scale
        )
        total_loss = main_loss

        if aux_list and (self.use_bce or self.use_dice):
            aux_total = main_logits.new_zeros(())
            for w, aux_logits in zip(self.aux_weights, aux_list):
                if w <= 0:
                    continue
                l_aux, _ = self._single_scale_loss(
                    aux_logits, targets, include_topo=False, topo_scale=0.0
                )
                aux_total = aux_total + w * l_aux
            total_loss = total_loss + aux_total
            stats["aux"] = float(aux_total.detach().item())

        stats["loss"] = float(total_loss.detach().item())
        return total_loss, stats
