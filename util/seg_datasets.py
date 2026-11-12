import json
import math
import os
import random
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


DATASET_RULES = {
    "drive": {
        "img_regex": re.compile(r"^(?P<prefix>.*?)(?P<id>\d{2})_(?:training|test)\.tif$"),
        "lbl_fmt": "{prefix}{id}_manual1.png",
        "default_val_groups": 4,
    },
    "chase": {
        "img_regex": re.compile(r"^(?P<prefix>.*?)Image_(?P<id>\d{2})(?P<eye>[LR])\.jpg$"),
        "lbl_fmt": "{prefix}Image_{id}{eye}_1stHO.png",
        "default_val_groups": 2,
    },
    "stare": {
        "img_regex": re.compile(r"^(?P<prefix>.*?)im(?P<id>\d{4})\.png$"),
        "lbl_fmt": "{prefix}im{id}.ah.png",
        "default_val_groups": 3,
    },
}


def infer_dataset_name(data_path: str, explicit_name: str = "") -> str:
    if explicit_name and explicit_name.lower() in DATASET_RULES:
        return explicit_name.lower()
    base = os.path.basename(os.path.normpath(data_path)).lower()
    for key in DATASET_RULES:
        if key in base:
            return key
    raise ValueError(
        f"Cannot infer dataset type from path '{data_path}'. "
        f"Please pass --dataset one of {list(DATASET_RULES.keys())}."
    )


def pair_split_files(data_path: str, split: str, dataset_name: str) -> List[Dict[str, object]]:
    """Pair images and labels in `{data_path}/{split}/{images,labels}` and extract group IDs."""
    rules = DATASET_RULES[dataset_name]
    img_dir = os.path.join(data_path, split, "images")
    lbl_dir = os.path.join(data_path, split, "labels")
    if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
        raise FileNotFoundError(f"Expected directories '{img_dir}' and '{lbl_dir}'")

    records: List[Dict[str, object]] = []
    for fname in sorted(os.listdir(img_dir)):
        if fname.startswith("."):
            continue
        img_path = os.path.join(img_dir, fname)
        if not os.path.isfile(img_path):
            continue
        m = rules["img_regex"].match(fname)
        if m is None:
            continue
        gd = m.groupdict()
        lbl_name = rules["lbl_fmt"].format(**gd)
        lbl_path = os.path.join(lbl_dir, lbl_name)
        if not os.path.isfile(lbl_path):
            raise FileNotFoundError(
                f"Missing paired label '{lbl_path}' for image '{img_path}'"
            )

        prefix = gd.get("prefix", "")
        group_id = gd["id"]
        is_original = prefix == ""
        records.append(
            {
                "name": os.path.splitext(fname)[0],
                "img_name": fname,
                "lbl_name": lbl_name,
                "img_path": img_path,
                "lbl_path": lbl_path,
                "group_id": group_id,
                "prefix": prefix,
                "is_original": bool(is_original),
            }
        )

    if not records:
        raise RuntimeError(f"No valid image-label pairs found in {img_dir}")
    return records


def build_group_aware_train_val_split(
    data_path: str,
    dataset_name: str,
    n_val_groups: Optional[int] = None,
    val_ids: Optional[Sequence[str]] = None,
    datasets_seed: int = 2026,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    """Split `train/` records by source group_id with zero data leakage.

    - All augmented variants of held-out `val_ids` are excluded from training.
    - Validation uses only the unaugmented (`is_original=True`) images of the held-out
      source IDs so validation reflects clean, un-warped test images.
    """
    all_train = pair_split_files(data_path, "train", dataset_name)
    unique_groups = sorted({str(r["group_id"]) for r in all_train})

    if val_ids:
        chosen_val = sorted({str(v).strip() for v in val_ids if str(v).strip()})
        unknown = [v for v in chosen_val if v not in unique_groups]
        if unknown:
            raise ValueError(
                f"Requested val_ids {unknown} not in available groups {unique_groups}"
            )
    else:
        if n_val_groups is None or n_val_groups <= 0:
            n_val_groups = DATASET_RULES[dataset_name]["default_val_groups"]
        rng = random.Random(int(datasets_seed))
        groups_copy = list(unique_groups)
        rng.shuffle(groups_copy)
        chosen_val = sorted(groups_copy[: int(n_val_groups)])

    val_set = set(chosen_val)
    train_records = [r for r in all_train if str(r["group_id"]) not in val_set]
    val_records = [
        r for r in all_train if str(r["group_id"]) in val_set and bool(r["is_original"])
    ]

    # Safety assertions: zero overlap in group_id
    train_groups = {str(r["group_id"]) for r in train_records}
    val_groups = {str(r["group_id"]) for r in val_records}
    assert train_groups.isdisjoint(val_groups), (
        f"Data leakage detected between train {train_groups} and val {val_groups}"
    )

    split_info = {
        "dataset": dataset_name,
        "datasets_seed": int(datasets_seed),
        "all_groups": unique_groups,
        "train_groups": sorted(train_groups),
        "val_groups": sorted(val_groups),
        "num_train_files": len(train_records),
        "num_val_files": len(val_records),
    }
    return train_records, val_records, split_info


def compute_fov_mask(rgb: np.ndarray, threshold: int = 15, erode_px: int = 2) -> np.ndarray:
    """Derive binary Field-of-View (FOV) mask from an RGB fundus image (uint8 HxWx3).

    Steps:
      1. Take max across RGB channels (or red/green) and smooth with 5x5 Gaussian.
      2. Threshold > `threshold`.
      3. Keep largest connected component.
      4. Fill internal holes via corner flood-fill.
      5. Erode by `erode_px` pixels to exclude bright aperture boundary ring.
    """
    if rgb.ndim == 3:
        intensity = rgb.max(axis=2).astype(np.uint8)
    else:
        intensity = rgb.astype(np.uint8)

    blurred = cv2.GaussianBlur(intensity, (5, 5), 0)
    binary = (blurred > threshold).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels > 1:
        # Label 0 is background; pick largest foreground component
        fg_areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = 1 + int(np.argmax(fg_areas))
        mask = (labels == largest_label).astype(np.uint8)
    else:
        mask = binary

    # Fill internal holes by flood-filling background from padded border
    h, w = mask.shape
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = mask * 255
    inv = padded.copy()
    cv2.floodFill(inv, None, (0, 0), 255)
    holes = (inv[1:-1, 1:-1] == 0).astype(np.uint8)
    filled = np.logical_or(mask > 0, holes > 0).astype(np.uint8)

    if erode_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        filled = cv2.erode(filled, kernel, iterations=int(erode_px))

    return filled.astype(np.uint8)


def apply_clahe_rgb(rgb: np.ndarray) -> np.ndarray:
    """Optional CLAHE contrast enhancement on the LAB luminance / green channel."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enh = clahe.apply(l_ch)
    merged = cv2.merge((l_enh, a_ch, b_ch))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def pad_to_multiple(
    tensor: torch.Tensor, multiple: int = 16, mode: str = "reflect"
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad `(..., H, W)` tensor on bottom and right so H, W are divisible by `multiple`."""
    orig_h, orig_w = tensor.shape[-2], tensor.shape[-1]
    pad_h = (multiple - (orig_h % multiple)) % multiple
    pad_w = (multiple - (orig_w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, (orig_h, orig_w)
    if tensor.ndim == 3:
        padded = F.pad(tensor.unsqueeze(0), (0, pad_w, 0, pad_h), mode=mode).squeeze(0)
    else:
        padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)
    return padded, (orig_h, orig_w)


import torch.nn.functional as F


class VesselSegDataset(Dataset):
    """Retinal vessel segmentation dataset supporting multi-crop training and full-image eval."""

    def __init__(
        self,
        records: List[Dict[str, object]],
        is_train: bool = True,
        crop_size: int = 448,
        crops_per_image: int = 2,
        enhance: bool = False,
        mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
        std: Sequence[float] = IMAGENET_DEFAULT_STD,
    ):
        super().__init__()
        self.records = list(records)
        self.is_train = bool(is_train)
        self.crop_size = int(crop_size)
        self.crops_per_image = int(crops_per_image) if is_train else 1
        self.enhance = bool(enhance)
        self.mean = list(mean)
        self.std = list(std)

        # Cache decoded images, binary masks, and FOV masks in RAM (datasets are <=260 images)
        self._cache: List[Dict[str, object]] = []
        for rec in self.records:
            img_pil = Image.open(rec["img_path"]).convert("RGB")
            lbl_pil = Image.open(rec["lbl_path"]).convert("L")
            rgb = np.array(img_pil, dtype=np.uint8)
            lbl = (np.array(lbl_pil) > 0).astype(np.uint8)
            fov = compute_fov_mask(rgb)
            if self.enhance:
                rgb_proc = apply_clahe_rgb(rgb)
            else:
                rgb_proc = rgb

            self._cache.append(
                {
                    "name": rec["name"],
                    "group_id": rec["group_id"],
                    "rgb": rgb_proc,
                    "lbl": lbl,
                    "fov": fov,
                }
            )

    def __len__(self) -> int:
        return len(self._cache)

    def _augment_one_crop(
        self, rgb: np.ndarray, lbl: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        H, W, _ = rgb.shape
        cs = self.crop_size

        # Pad if image is smaller than crop_size
        if H < cs or W < cs:
            pad_h = max(0, cs - H)
            pad_w = max(0, cs - W)
            rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            lbl = np.pad(lbl, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
            H, W, _ = rgb.shape

        # Random crop
        top = random.randint(0, H - cs)
        left = random.randint(0, W - cs)
        rgb_crop = rgb[top : top + cs, left : left + cs]
        lbl_crop = lbl[top : top + cs, left : left + cs]

        # Random horizontal & vertical flips
        if random.random() < 0.5:
            rgb_crop = np.ascontiguousarray(rgb_crop[:, ::-1])
            lbl_crop = np.ascontiguousarray(lbl_crop[:, ::-1])
        if random.random() < 0.5:
            rgb_crop = np.ascontiguousarray(rgb_crop[::-1, :])
            lbl_crop = np.ascontiguousarray(lbl_crop[::-1, :])

        # Random 90-degree rotations
        k90 = random.randint(0, 3)
        if k90 > 0:
            rgb_crop = np.ascontiguousarray(np.rot90(rgb_crop, k90))
            lbl_crop = np.ascontiguousarray(np.rot90(lbl_crop, k90))

        # Random small rotation (-15 to +15 deg) with probability 0.5
        if random.random() < 0.5:
            angle = random.uniform(-15.0, 15.0)
            center = (cs * 0.5, cs * 0.5)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            rgb_crop = cv2.warpAffine(
                rgb_crop,
                rot_mat,
                (cs, cs),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            lbl_crop = cv2.warpAffine(
                lbl_crop,
                rot_mat,
                (cs, cs),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        img_t = TF.to_tensor(rgb_crop)  # float32 in [0, 1], shape (3, cs, cs)

        # Light photometric jitter (brightness, contrast, saturation, gamma)
        if random.random() < 0.8:
            b_factor = random.uniform(0.9, 1.1)
            c_factor = random.uniform(0.9, 1.1)
            s_factor = random.uniform(0.9, 1.1)
            gamma = random.uniform(0.85, 1.15)
            img_t = TF.adjust_brightness(img_t, b_factor)
            img_t = TF.adjust_contrast(img_t, c_factor)
            img_t = TF.adjust_saturation(img_t, s_factor)
            img_t = TF.adjust_gamma(img_t.clamp(0.0, 1.0), gamma)

        img_t = TF.normalize(img_t, mean=self.mean, std=self.std)
        lbl_t = torch.from_numpy((lbl_crop > 0).astype(np.float32)).unsqueeze(0)
        return img_t, lbl_t

    def __getitem__(self, idx: int):
        item = self._cache[idx]
        rgb = item["rgb"]
        lbl = item["lbl"]
        fov = item["fov"]

        if self.is_train:
            imgs = []
            lbls = []
            for _ in range(self.crops_per_image):
                im_c, lb_c = self._augment_one_crop(rgb, lbl)
                imgs.append(im_c)
                lbls.append(lb_c)
            # Shape: (K, 3, cs, cs) and (K, 1, cs, cs)
            return torch.stack(imgs, dim=0), torch.stack(lbls, dim=0)

        img_t = TF.normalize(TF.to_tensor(rgb), mean=self.mean, std=self.std)
        lbl_t = torch.from_numpy((lbl > 0).astype(np.float32)).unsqueeze(0)
        fov_t = torch.from_numpy((fov > 0).astype(np.uint8))
        return {
            "image": img_t,
            "label": lbl_t,
            "fov": fov_t,
            "name": item["name"],
            "group_id": item["group_id"],
        }


def collate_multi_crop(batch):
    """Flatten `(B, K, C, H, W)` from multi-crop training into `(B*K, C, H, W)`."""
    imgs = torch.cat([b[0] for b in batch], dim=0)
    lbls = torch.cat([b[1] for b in batch], dim=0)
    return imgs, lbls
