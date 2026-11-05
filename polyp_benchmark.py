"""Reproducible, local six-model polyp segmentation benchmark."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import types

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
BASE = Path("/home/Data_Drive_8TB_2/kykim/polyp/benchmark")
MODELS = ("unet", "pranet", "hardnet_mseg", "dinov2", "segformer", "polyp_pvt")
LABELS = {
    "unet": "U-Net (base 32)",
    "pranet": "PraNet / Res2Net-50",
    "hardnet_mseg": "HarDNet-MSEG / HarDNet-68",
    "dinov2": "DINOv2 ViT-S/14 + multilevel decoder",
    "segformer": "SegFormer / MiT-B2",
    "polyp_pvt": "Polyp-PVT / PVTv2-B2",
}
SOURCES = {
    "PraNet": ("DengPingFan/PraNet", "95f12bd4e32ec954b5069654abae7de9473fd1f5"),
    "Polyp-PVT": ("DengPingFan/Polyp-PVT", "cd71fbf9fcd8c32cc0f33699f4e0186073c64dc4"),
    "HarDNet-MSEG": ("james128333/HarDNet-MSEG", "aac1c4b5d2fdbc80bc8fe21a5377af2d202a0348"),
}
WEIGHTS = {
    "pranet": "https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/"
    "res2net50_v1b_26w_4s-3cf99910.pth",
    "hardnet_mseg": "https://ping-chao.com/hardnet/hardnet68-5d684880.pth",
    "polyp_pvt": "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth",
}


@dataclass(frozen=True)
class Config:
    data: str = "/home/Data_Drive_8TB_2/kykim/polyp/data"
    output: str = str(BASE / "seed42_448")
    gpu: str = "1"
    seed: int = 42
    size: int = 448
    batch_size: int = 4
    accumulation: int = 2
    epochs: int = 50
    patience: int = 10
    min_delta: float = 0.0001
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    workers: int = 4
    threshold: float = 0.5
    val_fraction: float = 0.2
    pretrained: bool = True
    rle_order: str = "C"


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another benchmark is running: {path}") from exc
        yield


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(0)


def device_for(cfg: Config) -> torch.device:
    if "torch" in sys.modules and torch.cuda.is_initialized():
        if os.environ.get("CUDA_VISIBLE_DEVICES") != cfg.gpu:
            raise RuntimeError("CUDA already initialized on another GPU; use a fresh subprocess.")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the approved experiment.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This benchmark requires BF16 support on the selected GPU.")
    return torch.device("cuda:0")


def prepare_sources() -> dict:
    """Keep the upstream architectures; patch only imports and implicit weight loading."""
    manifest = {}
    patches = {
        "PraNet": [
            ("lib/PraNet_Res2Net.py",
             "self.resnet = res2net50_v1b_26w_4s(pretrained=True)",
             "self.resnet = res2net50_v1b_26w_4s(pretrained=False)"),
        ],
        "HarDNet-MSEG": [
            ("lib/HarDMSEG.py", "self.hardnet = hardnet(arch=68)",
             "self.hardnet = hardnet(arch=68, pretrained=False)"),
        ],
        "Polyp-PVT": [
            ("lib/pvt.py", "from lib.pvtv2 import pvt_v2_b2",
             "from .pvtv2 import pvt_v2_b2"),
            ("lib/pvt.py",
             "        path = './pretrained_pth/pvt_v2_b2.pth'\n"
             "        save_model = torch.load(path)\n"
             "        model_dict = self.backbone.state_dict()\n"
             "        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}\n"
             "        model_dict.update(state_dict)\n"
             "        self.backbone.load_state_dict(model_dict)",
             "        # Generic backbone weights are loaded explicitly by the benchmark."),
        ],
    }
    for name, (repo, revision) in SOURCES.items():
        dest = BASE / "vendor" / name
        if not dest.exists():
            subprocess.run(["git", "clone", f"https://github.com/{repo}.git", str(dest)],
                           check=True)
            subprocess.run(["git", "-C", str(dest), "checkout", "--detach", revision],
                           check=True)
        actual = subprocess.check_output(
            ["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise RuntimeError(f"Unexpected source revision for {name}: {actual}")
        allowed = {item[0] for item in patches[name]}
        changed = subprocess.check_output(
            ["git", "-C", str(dest), "diff", "--name-only"], text=True).splitlines()
        if set(changed) - allowed:
            raise RuntimeError(f"Unrecognized upstream modifications in {name}: {changed}")
        for relative, old, new in patches[name]:
            path = dest / relative
            text = path.read_text()
            if old in text:
                if text.count(old) != 1:
                    raise RuntimeError(f"Ambiguous compatibility patch: {path}")
                path.write_text(text.replace(old, new))
            elif new not in text:
                raise RuntimeError(f"Compatibility patch does not match: {path}")
        diff = subprocess.check_output(
            ["git", "-C", str(dest), "diff", "--no-ext-diff"], text=True)
        manifest[name] = {
            "repository": f"https://github.com/{repo}", "revision": actual,
            "compatibility_diff": diff,
            "python_sha256": {str(p.relative_to(dest)): file_hash(p)
                              for p in sorted((dest / "lib").glob("*.py"))},
        }
    atomic_json(BASE / "sources.json", manifest)
    return manifest


def load_official(repo: str, module: str):
    package_name = "_polyp_" + repo.replace("-", "_").lower()
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(BASE / "vendor" / repo / "lib")]
        sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.{module}")


def generic_weights(name: str) -> dict:
    url = WEIGHTS[name]
    return torch.hub.load_state_dict_from_url(
        url, model_dir=str(BASE / "weights"), map_location="cpu",
        check_hash=name != "polyp_pvt", weights_only=True)


class DoubleConv(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True), nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        channels = [base * 2**i for i in range(5)]
        self.encoder = nn.ModuleList(
            [DoubleConv(3, channels[0])] +
            [DoubleConv(a, b) for a, b in zip(channels, channels[1:])])
        self.up = nn.ModuleList(
            nn.ConvTranspose2d(a, b, 2, stride=2)
            for a, b in zip(channels[:0:-1], channels[-2::-1]))
        self.decoder = nn.ModuleList(
            DoubleConv(2 * b, b) for b in channels[-2::-1])
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.encoder):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        for up, block, skip in zip(self.up, self.decoder, skips[-2::-1]):
            x = up(x)
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat((x, skip), dim=1))
        return self.head(x)


class DinoSegmentation(nn.Module):
    def __init__(self, size: int, pretrained: bool):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m", pretrained=pretrained,
            img_size=size, num_classes=0, dynamic_img_size=True)
        self.projections = nn.ModuleList(nn.Conv2d(384, 96, 1) for _ in range(4))
        self.fuse = DoubleConv(384, 128)
        self.refine = DoubleConv(128, 64)
        self.head = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        features = self.backbone.forward_intermediates(
            x, indices=(2, 5, 8, 11), intermediates_only=True, norm=True,
            output_fmt="NCHW")
        z = self.fuse(torch.cat(
            [proj(f) for proj, f in zip(self.projections, features)], dim=1))
        z = F.interpolate(z, scale_factor=4, mode="bilinear", align_corners=False)
        return F.interpolate(self.head(self.refine(z)), size=x.shape[-2:],
                             mode="bilinear", align_corners=False)


class SegFormer(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerModel
        if pretrained:
            backbone = SegformerModel.from_pretrained("nvidia/mit-b2")
            config = backbone.config
        else:
            backbone = None
            config = SegformerConfig(
                depths=[3, 4, 6, 3], hidden_sizes=[64, 128, 320, 512],
                num_attention_heads=[1, 2, 5, 8], decoder_hidden_size=768)
        config.num_labels = 1
        config.decoder_hidden_size = 768
        self.network = SegformerForSemanticSegmentation(config)
        if backbone is not None:
            self.network.segformer = backbone

    def forward(self, x):
        return F.interpolate(self.network(pixel_values=x).logits, size=x.shape[-2:],
                             mode="bilinear", align_corners=False)


def build_model(name: str, cfg: Config) -> nn.Module:
    if name == "unet":
        return UNet()
    if name == "dinov2":
        return DinoSegmentation(cfg.size, cfg.pretrained)
    if name == "segformer":
        return SegFormer(cfg.pretrained)
    if name == "pranet":
        model = load_official("PraNet", "PraNet_Res2Net").PraNet()
        backbone = model.resnet
    elif name == "hardnet_mseg":
        model = load_official("HarDNet-MSEG", "HarDMSEG").HarDMSEG()
        backbone = model.hardnet
    elif name == "polyp_pvt":
        model = load_official("Polyp-PVT", "pvt").PolypPVT()
        backbone = model.backbone
    else:
        raise ValueError(f"Unknown model: {name}")
    if cfg.pretrained:
        weights = generic_weights(name)
        if name == "polyp_pvt":
            # The official segmentation encoder omits the ImageNet classifier.
            if weights["head.weight"].shape != (1000, 512) or weights["head.bias"].shape != (1000,):
                raise ValueError("Unexpected PVT ImageNet classification head.")
            del weights["head.weight"], weights["head.bias"]
        backbone.load_state_dict(weights, strict=True)
    return model


def outputs_of(output) -> list[torch.Tensor]:
    return list(output) if isinstance(output, (tuple, list)) else [output]


def final_logit(output) -> torch.Tensor:
    return outputs_of(output)[-1]


def segmentation_loss(output, target: torch.Tensor) -> torch.Tensor:
    losses = []
    for logit in outputs_of(output):
        logit = F.interpolate(logit.float(), size=target.shape[-2:],
                              mode="bilinear", align_corners=False)
        prob = logit.sigmoid()
        intersection = (prob * target).sum((1, 2, 3))
        dice = (2 * intersection + 1) / (prob.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
        losses.append(F.binary_cross_entropy_with_logits(logit, target) + 1 - dice.mean())
    return torch.stack(losses).mean()


def rle_encode(mask: np.ndarray, order: str = "C") -> str:
    if mask.ndim != 2 or not np.isin(mask, [0, 1]).all():
        raise ValueError("RLE requires a two-dimensional binary mask.")
    flat = np.asarray(mask, dtype=np.uint8).ravel(order=order)
    padded = np.concatenate(([0], flat, [0]))
    changes = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    changes[1::2] -= changes[::2]
    return " ".join(map(str, changes))


def rle_decode(value: str, shape: tuple[int, int], order: str = "C") -> np.ndarray:
    parts = np.array([int(t) for t in value.split()], dtype=np.int64)
    if parts.size % 2:
        raise ValueError("RLE must contain start/length pairs.")
    pairs = parts.reshape(-1, 2)
    flat = np.zeros(math.prod(shape), dtype=np.uint8)
    end = 0
    for start, length in pairs:
        if start < 1 or length < 1 or start - 1 < end or start - 1 + length > flat.size:
            raise ValueError("Invalid or overlapping RLE run.")
        end = start - 1 + length
        flat[start - 1:end] = 1
    return flat.reshape(shape, order=order)


def mask_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    pred, truth = pred.astype(bool), truth.astype(bool)
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    precision = tp / (tp + fp) if tp + fp else float(not truth.any())
    recall = tp / (tp + fn) if tp + fn else 1.0
    a = pred.astype(np.uint8)
    b = truth.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    ba = a - cv2.erode(a, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    bb = b - cv2.erode(b, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    tolerance = max(1, round(0.005 * math.hypot(*truth.shape)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance + 1,) * 2)
    pa = float((ba * cv2.dilate(bb, k)).sum() / ba.sum()) if ba.any() else float(not bb.any())
    rb = float((bb * cv2.dilate(ba, k)).sum() / bb.sum()) if bb.any() else float(not ba.any())
    boundary = 2 * pa * rb / (pa + rb) if pa + rb else 0.0
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall,
            "boundary_f1": boundary, "area_fraction": float(truth.mean()),
            "tp": tp, "fp": fp, "fn": fn}


def inspect_data(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = Path(cfg.data)
    images = sorted((data / "Dataset/images").glob("*.png"))
    masks = sorted((data / "Dataset/masks").glob("*.png"))
    if len(images) != 800 or {p.name for p in images} != {p.name for p in masks}:
        raise ValueError("Expected 800 exactly paired training images and masks.")
    test = sorted((data / "Test/images").glob("*.png"))
    sample = pd.read_csv(data / "sample_submission.csv", keep_default_na=False)
    if list(sample.columns) != ["id", "target"] or len(sample) != 200 or not sample.id.is_unique:
        raise ValueError("Unexpected sample submission schema.")
    expected_names = {f"{int(i):03}.png" for i in sample.id}
    if {p.name for p in test} != expected_names:
        raise ValueError("Test filenames do not match sample submission IDs.")
    rows = []
    for path in images:
        mask_path = data / "Dataset/masks" / path.name
        with Image.open(path) as im, Image.open(mask_path) as m:
            if im.size != m.size:
                raise ValueError(f"Image/mask dimensions differ: {path.name}")
            a = np.asarray(m.convert("L"))
            if a.min() >= 128 or a.max() <= 127:
                raise ValueError(f"Annotation lacks foreground or background: {mask_path}")
            rows.append({"name": path.name, "width": im.width, "height": im.height,
                         "area_fraction": float((a > 127).mean()),
                         "nonbinary_fraction": float((~np.isin(a, [0, 255])).mean()),
                         "image_sha256": file_hash(path), "mask_sha256": file_hash(mask_path)})
    frame = pd.DataFrame(rows)
    if frame.image_sha256.duplicated().any():
        raise ValueError("Duplicate training images require grouped splitting.")
    hashes = set(frame.image_sha256)
    if any(file_hash(p) in hashes for p in test):
        raise ValueError("Identical train/test image detected.")
    # Target strings in sample_submission are never training/validation labels.
    rng = np.random.default_rng(cfg.seed)
    permutation = rng.permutation(len(frame))
    val_indices = set(permutation[:round(len(frame) * cfg.val_fraction)].tolist())
    frame["split"] = ["val" if i in val_indices else "train" for i in range(len(frame))]
    return frame, sample


def initialize_run(cfg: Config) -> pd.DataFrame:
    out = Path(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    if cfg.size % 224 or cfg.rle_order != "C":
        raise ValueError("Shared size must be divisible by 32 and 14; competition RLE is row-major.")
    expected = {"config": asdict(cfg), "runner_sha256": file_hash(Path(__file__))}
    config_path = out / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != expected:
            raise RuntimeError("Run configuration/code changed; select a new output directory.")
        frame = pd.read_csv(out / "split.csv")
        return frame
    frame, sample = inspect_data(cfg)
    frame.to_csv(out / "split.csv", index=False)
    test_rows = []
    for identifier in sample.id:
        path = Path(cfg.data) / "Test/images" / f"{int(identifier):03}.png"
        with Image.open(path) as im:
            test_rows.append({"id": int(identifier), "name": path.name,
                              "width": im.width, "height": im.height,
                              "image_sha256": file_hash(path)})
    pd.DataFrame(test_rows).to_csv(out / "test_manifest.csv", index=False)
    atomic_json(config_path, expected)
    return frame


class PolypDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, cfg: Config, training: bool):
        self.frame = frame.reset_index(drop=True)
        self.cfg = cfg
        self.training = training
        self.images, self.masks = [], []
        for name in self.frame.name:
            with Image.open(Path(cfg.data) / "Dataset/images" / name) as im:
                self.images.append(np.array(im.convert("RGB").resize(
                    (cfg.size, cfg.size), Image.Resampling.BILINEAR)))
            with Image.open(Path(cfg.data) / "Dataset/masks" / name) as m:
                self.masks.append(np.array(m.convert("L").resize(
                    (cfg.size, cfg.size), Image.Resampling.NEAREST)))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        im, mask = self.images[index], self.masks[index]
        if self.training:
            turns = random.randrange(4)
            im, mask = np.rot90(im, turns), np.rot90(mask, turns)
            if random.random() < 0.5:
                im, mask = np.fliplr(im), np.fliplr(mask)
            if random.random() < 0.5:
                im, mask = np.flipud(im), np.flipud(mask)
            image = Image.fromarray(np.ascontiguousarray(im))
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
            im = np.array(ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15)))
        x = torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))).float() / 255
        x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor(
            [0.229, 0.224, 0.225])[:, None, None]
        y = torch.from_numpy(np.ascontiguousarray(mask > 127)[None]).float()
        return x, y, index


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    cv2.setNumThreads(0)


def make_loader(dataset: PolypDataset, cfg: Config, training: bool, epoch: int = 0):
    generator = torch.Generator().manual_seed(cfg.seed + epoch)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=training,
                      num_workers=cfg.workers, pin_memory=True, worker_init_fn=worker_seed,
                      generator=generator, persistent_workers=False)


@torch.inference_mode()
def evaluate(model: nn.Module, dataset: PolypDataset, cfg: Config, device,
             save_dir: Path | None = None) -> pd.DataFrame:
    model.eval()
    rows = []
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    for x, _, indices in make_loader(dataset, cfg, False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = final_logit(model(x.to(device)))
        for z, index in zip(logits.float(), indices.tolist()):
            name = dataset.frame.iloc[index]["name"]
            with Image.open(Path(cfg.data) / "Dataset/masks" / name) as m:
                truth = np.array(m.convert("L")) > 127
            # Resize probabilities, not hard labels; exactly the same path is used for test.
            prob = F.interpolate(z[None].sigmoid(), size=truth.shape, mode="bilinear",
                                 align_corners=False)[0, 0].cpu().numpy()
            pred = prob >= cfg.threshold
            rows.append({"name": name, **mask_metrics(pred, truth)})
            if save_dir is not None:
                Image.fromarray(pred.astype(np.uint8) * 255).save(save_dir / name)
    result = pd.DataFrame(rows)
    if len(result) != len(dataset) or not np.isfinite(result.select_dtypes("number")).all().all():
        raise RuntimeError("Invalid validation metrics.")
    return result


@torch.inference_mode()
def write_submission(model, cfg: Config, device, model_dir: Path) -> Path:
    model.eval()
    data = Path(cfg.data)
    sample = pd.read_csv(data / "sample_submission.csv", keep_default_na=False)
    targets = []
    for identifier in sample.id:
        with Image.open(data / "Test/images" / f"{int(identifier):03}.png") as im:
            original_shape = (im.height, im.width)
            a = np.array(im.convert("RGB").resize((cfg.size, cfg.size), Image.Resampling.BILINEAR))
        x = torch.from_numpy(a.transpose(2, 0, 1).copy()).float() / 255
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = final_logit(model(((x - mean) / std)[None].to(device))).float()
        prob = F.interpolate(z.sigmoid(), size=original_shape, mode="bilinear",
                             align_corners=False)[0, 0].cpu().numpy()
        pred = (prob >= cfg.threshold).astype(np.uint8)
        encoded = rle_encode(pred, cfg.rle_order)
        if not np.array_equal(rle_decode(encoded, original_shape, cfg.rle_order), pred):
            raise RuntimeError(f"RLE round trip failed for test ID {identifier}")
        targets.append(encoded)
    result = pd.DataFrame({"id": sample.id, "target": targets})
    path = model_dir / "submission.csv"
    result.to_csv(path, index=False)
    persisted = pd.read_csv(path, keep_default_na=False)
    if not persisted.equals(result):
        raise RuntimeError("Submission changed during CSV serialization.")
    return path


def save_checkpoint(path: Path, value: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


@torch.inference_mode()
def benchmark_latency(model, cfg: Config, device) -> float:
    model.eval()
    x = torch.zeros(1, 3, cfg.size, cfg.size, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(5):
            model(x)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            model(x)
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - start) / 20


def train_model(name: str, cfg: Config) -> dict:
    device = device_for(cfg)
    seed_everything(cfg.seed)
    out = Path(cfg.output)
    model_dir = out / name
    model_dir.mkdir(parents=True, exist_ok=True)
    status_path = model_dir / "status.json"
    with exclusive_lock(BASE / f"gpu{cfg.gpu}.lock"):
        frame = initialize_run(cfg)
        result_path = model_dir / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            csv_path = Path(result["submission"])
            if not csv_path.exists() or file_hash(csv_path) != result["submission_sha256"]:
                raise RuntimeError(f"Completed submission is missing/modified: {csv_path}")
            print(f"{name}: already completed", flush=True)
            return result
        atomic_json(status_path, {"state": "starting", "model": name, "pid": os.getpid()})
        try:
            return _train_model(name, cfg, device, frame, model_dir)
        except Exception as exc:
            atomic_json(status_path, {"state": "failed", "model": name,
                                     "error": str(exc), "traceback": traceback.format_exc()})
            raise


def _train_model(name, cfg, device, frame, model_dir):
    model = build_model(name, cfg).to(device)
    train_set = PolypDataset(frame[frame.split == "train"], cfg, True)
    val_set = PolypDataset(frame[frame.split == "val"], cfg, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                 weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    start_epoch, best, best_epoch, stale, elapsed = 0, -1.0, 0, 0, 0.0
    history = []
    last_path = model_dir / "last.pt"
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, best, best_epoch, stale = (
            checkpoint[k] for k in ("epoch", "best", "best_epoch", "stale"))
        history, elapsed = checkpoint["history"], checkpoint["elapsed"]
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, cfg.epochs):
        if stale >= cfg.patience:
            break
        tick = time.perf_counter()
        model.train()
        loader = make_loader(train_set, cfg, True, epoch)
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, (x, y, _) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            group_start = (step // cfg.accumulation) * cfg.accumulation
            group_size = min(cfg.accumulation, len(loader) - group_start)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = segmentation_loss(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss: {name}, epoch {epoch + 1}, step {step}")
            scaler.scale(loss / group_size).backward()
            if (step + 1) % cfg.accumulation == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach()) * len(x)
        metrics = evaluate(model, val_set, cfg, device)
        score = float(metrics.dice.mean())
        scheduler.step()
        seconds = time.perf_counter() - tick
        elapsed += seconds
        row = {"epoch": epoch + 1, "train_loss": running / len(train_set),
               "val_dice": score, "val_iou": float(metrics.iou.mean()),
               "seconds": seconds, "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        # Always preserve the actual maximum; min_delta controls only patience.
        significant = score > best + cfg.min_delta
        if score > best:
            best, best_epoch = score, epoch + 1
            save_checkpoint(model_dir / "best.pt", {"model": model.state_dict(),
                                                   "epoch": best_epoch, "dice": best})
        stale = 0 if significant else stale + 1
        pd.DataFrame(history).to_csv(model_dir / "history.csv", index=False)
        save_checkpoint(last_path, {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch + 1, "best": best, "best_epoch": best_epoch, "stale": stale,
            "history": history, "elapsed": elapsed, "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()})
        atomic_json(model_dir / "status.json", {"state": "training", "model": name,
                    "epoch": epoch + 1, "max_epochs": cfg.epochs, "best_dice": best,
                    "best_epoch": best_epoch, "pid": os.getpid(), "last_epoch_seconds": seconds})
        print(f"{name}: epoch {epoch+1}/{cfg.epochs}, loss={row['train_loss']:.4f}, "
              f"native Dice={score:.5f}, best={best:.5f}, {seconds:.1f}s", flush=True)
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    checkpoint = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    atomic_json(model_dir / "status.json", {"state": "evaluating_and_encoding", "model": name})
    metrics = evaluate(model, val_set, cfg, device, model_dir / "val_masks")
    metrics.to_csv(model_dir / "val_metrics.csv", index=False)
    submission = write_submission(model, cfg, device, model_dir)
    tp, fp, fn = metrics[["tp", "fp", "fn"]].sum()
    result = {"model": name, "label": LABELS[name],
              "family": "CNN" if name in MODELS[:3] else "Transformer",
              "best_epoch": best_epoch, "epochs_run": len(history),
              **{f"val_{k}": float(metrics[k].mean())
                 for k in ("dice", "iou", "precision", "recall", "boundary_f1")},
              "val_global_dice": float(2 * tp / (2 * tp + fp + fn)),
              "parameters_m": sum(p.numel() for p in model.parameters()) / 1e6,
              "training_seconds": elapsed, "peak_allocated_mb": peak_mb,
              "inference_ms_batch1": benchmark_latency(model, cfg, device),
              "submission": str(submission), "submission_sha256": file_hash(submission)}
    atomic_json(model_dir / "result.json", result)
    atomic_json(model_dir / "status.json", {"state": "completed", **result})
    return result


def smoke_model(name: str, cfg: Config) -> None:
    device = device_for(cfg)
    seed_everything(cfg.seed)
    model = build_model(name, cfg).to(device).train()
    batch = cfg.batch_size
    x = torch.randn(batch, 3, cfg.size, cfg.size, device=device)
    target = (torch.rand(batch, 1, cfg.size, cfg.size, device=device) > 0.5).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(x)
        loss = segmentation_loss(output, target)
    if any(z.shape != (batch, 1, cfg.size, cfg.size) for z in outputs_of(output)):
        raise AssertionError(f"Incorrect segmentation shape: {name}")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):
        raise AssertionError(f"Invalid gradients: {name}")
    nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    print(json.dumps({"model": name, "loss": float(loss.detach()),
                      "outputs": len(outputs_of(output)), "backward": "passed",
                      "peak_mb": torch.cuda.max_memory_allocated() / 1024**2}), flush=True)


def comparison(cfg: Config, require_all: bool = True) -> pd.DataFrame:
    results = [json.loads(p.read_text()) for name in MODELS
               if (p := Path(cfg.output) / name / "result.json").exists()]
    if require_all and len(results) != len(MODELS):
        raise RuntimeError(f"Only {len(results)}/6 models completed; do not declare a winner.")
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values("val_dice", ascending=False).reset_index(drop=True)


def analyze(cfg: Config) -> pd.DataFrame:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    table = comparison(cfg)
    out = Path(cfg.output)
    rng = np.random.default_rng(cfg.seed)
    per_image = {name: pd.read_csv(out / name / "val_metrics.csv").set_index("name").sort_index()
                 for name in MODELS}
    winner = table.iloc[0].model
    common = per_image[winner].index
    if any(not frame.index.equals(common) for frame in per_image.values()):
        raise RuntimeError("Models were evaluated on different validation images.")
    indices = rng.integers(0, len(common), size=(2000, len(common)))
    paired = []
    sizes = []
    for i, row in table.iterrows():
        metrics = per_image[row.model]
        values = metrics.dice.to_numpy()
        lo, hi = np.quantile(values[indices].mean(1), [0.025, 0.975])
        table.loc[i, "dice_ci_low"], table.loc[i, "dice_ci_high"] = lo, hi
        difference = per_image[winner].dice.to_numpy() - values
        dlo, dhi = np.quantile(difference[indices].mean(1), [0.025, 0.975])
        paired.append({"winner": winner, "compared_to": row.model,
                       "mean_difference": float(difference.mean()),
                       "paired_ci_low": float(dlo), "paired_ci_high": float(dhi)})
        for label, selection in [
                ("small_<=5%", metrics.area_fraction <= 0.05),
                ("medium_5-20%", (metrics.area_fraction > 0.05) & (metrics.area_fraction <= 0.2)),
                ("large_>20%", metrics.area_fraction > 0.2)]:
            subset = metrics.loc[selection]
            sizes.append({"model": row.model, "size_group": label, "n": len(subset),
                          "dice": float(subset.dice.mean()) if len(subset) else None,
                          "recall": float(subset.recall.mean()) if len(subset) else None})
    table.to_csv(out / "comparison.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "paired_bootstrap.csv", index=False)
    pd.DataFrame(sizes).to_csv(out / "size_analysis.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].barh(table.label[::-1], table.val_dice[::-1],
                 color=["#4878cf" if f == "CNN" else "#e89845" for f in table.family[::-1]])
    axes[0].set(xlim=(0, 1), xlabel="Native-resolution mean Dice (threshold 0.5)")
    for name in MODELS:
        history = pd.read_csv(out / name / "history.csv")
        axes[1].plot(history.epoch, history.val_dice, label=name)
        axes[2].plot(history.epoch, history.train_loss, label=name)
    axes[1].set(xlabel="Epoch", ylabel="Validation Dice")
    axes[2].set(xlabel="Epoch", ylabel="Training BCE + soft Dice loss")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # Fixed examples plus failures chosen once using the winning model.
    fixed = list(common[:3])
    worst = per_image[winner].sort_values("dice").index[:3].tolist()
    examples = list(dict.fromkeys(fixed + worst))
    fig, axes = plt.subplots(len(examples), 8, figsize=(24, 3 * len(examples)), squeeze=False)
    for r, name in enumerate(examples):
        with Image.open(Path(cfg.data) / "Dataset/images" / name) as im:
            image = np.array(im.convert("RGB"))
        with Image.open(Path(cfg.data) / "Dataset/masks" / name) as m:
            truth = np.array(m.convert("L")) > 127
        axes[r, 0].imshow(image)
        axes[r, 0].set_title(name[:14])
        axes[r, 1].imshow(truth, cmap="gray")
        axes[r, 1].set_title("Ground truth")
        for c, model_name in enumerate(MODELS, 2):
            with Image.open(out / model_name / "val_masks" / name) as m:
                pred = np.array(m) > 127
            overlay = image.astype(float) * 0.6
            overlay[pred & truth] += [0, 90, 0]
            overlay[pred & ~truth] += [100, 0, 0]
            overlay[~pred & truth] += [0, 0, 100]
            axes[r, c].imshow(np.clip(overlay, 0, 255).astype(np.uint8))
            axes[r, c].set_title(f"{model_name}\nDice={per_image[model_name].loc[name, 'dice']:.3f}")
        for ax in axes[r]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "qualitative_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    atomic_json(out / "analysis_summary.json", {
        "validation_winner": winner, "validation_dice": float(table.iloc[0].val_dice),
        "models_completed": len(table), "submitted_to_kaggle": False,
        "limitations": [
            "One image-level split, one seed; no patient IDs or external validation.",
            "Best epochs and winner selected on the same validation split: selection optimism.",
            "Bootstrap intervals are descriptive, image-level, and not corrected for winner selection.",
            "Different pretraining corpora and model capacities; not a causal CNN-vs-ViT test.",
            "No hidden-test labels or Kaggle leaderboard score were used.",
            "PraNet and Polyp-PVT upstream code is restricted to research/education.",
        ]})
    atomic_json(out / "status.json", {"state": "completed", "models_completed": 6,
                                     "winner": winner, "submitted": False})
    return table


def status(cfg: Config) -> pd.DataFrame:
    rows = []
    for name in MODELS:
        path = Path(cfg.output) / name / "status.json"
        rows.append(json.loads(path.read_text()) if path.exists()
                    else {"model": name, "state": "pending"})
    return pd.DataFrame(rows)


def prepare(cfg: Config) -> None:
    manifest = prepare_sources()
    frame = initialize_run(cfg)
    atomic_json(Path(cfg.output) / "sources.json", manifest)
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (Path(cfg.output) / "environment.txt").write_text(freeze)
    print(frame.groupby("split").size().to_string(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "smoke", "train", "analyze", "status"])
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--output", default=Config.output)
    args = parser.parse_args()
    cfg = Config(output=args.output)
    if args.command == "prepare":
        prepare(cfg)
    elif args.command in ("smoke", "train"):
        if not args.model:
            parser.error("--model is required")
        if args.command == "smoke":
            smoke_model(args.model, cfg)
        else:
            print(json.dumps(train_model(args.model, cfg), indent=2), flush=True)
    elif args.command == "analyze":
        print(analyze(cfg).to_string(index=False))
    else:
        print(status(cfg).to_string(index=False))


if __name__ == "__main__":
    main()
