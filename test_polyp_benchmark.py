from contextlib import nullcontext
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch

from polyp_benchmark import (
    Config, MODELS, UNet, analyze, atomic_json, comparison, exclusive_lock,
    mask_metrics, rle_decode, rle_encode, segmentation_loss, write_submission,
)


class EncodingTests(unittest.TestCase):
    def test_row_major_asymmetric_known_mask(self):
        mask = np.array([[0, 1, 1], [1, 0, 1]], dtype=np.uint8)
        self.assertEqual(rle_encode(mask), "2 3 6 1")
        np.testing.assert_array_equal(rle_decode("2 3 6 1", (2, 3)), mask)

    def test_empty_full_corners_and_random_roundtrips(self):
        rng = np.random.default_rng(42)
        masks = [np.zeros((3, 5), np.uint8), np.ones((3, 5), np.uint8),
                 np.eye(3, 5, dtype=np.uint8)]
        masks += [rng.integers(0, 2, (11, 17), dtype=np.uint8) for _ in range(10)]
        for mask in masks:
            with self.subTest(mask=mask.tolist()):
                np.testing.assert_array_equal(rle_decode(rle_encode(mask), mask.shape), mask)
        self.assertEqual(rle_encode(masks[0]), "")
        self.assertEqual(rle_encode(masks[1]), "1 15")

    def test_invalid_rle_is_rejected(self):
        for value in ["0 1", "1 0", "16 1", "1 3 2 2", "1", "-1 2"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                rle_decode(value, (3, 5))

    def test_invalid_mask_is_rejected(self):
        with self.assertRaises(ValueError):
            rle_encode(np.array([[0, 255]]))
        with self.assertRaises(ValueError):
            rle_encode(np.zeros(4))


class MetricTests(unittest.TestCase):
    def test_exact_metrics_and_boundary(self):
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[5:15, 5:20] = 1
        metric = mask_metrics(mask, mask)
        for key in ["dice", "iou", "precision", "recall", "boundary_f1"]:
            self.assertEqual(metric[key], 1)

    def test_empty_and_missed_polyp(self):
        zero = np.zeros((20, 30), np.uint8)
        truth = zero.copy()
        truth[5:15, 5:20] = 1
        self.assertEqual(mask_metrics(zero, zero)["dice"], 1)
        self.assertEqual(mask_metrics(zero, truth)["dice"], 0)
        self.assertEqual(mask_metrics(zero, truth)["recall"], 0)
        self.assertEqual(mask_metrics(truth, zero)["precision"], 0)

    def test_count_based_dice(self):
        pred = np.array([[1, 1], [0, 0]])
        truth = np.array([[1, 0], [1, 0]])
        metric = mask_metrics(pred, truth)
        self.assertEqual(metric["dice"], 0.5)
        self.assertAlmostEqual(metric["iou"], 1 / 3)


class TrainingTests(unittest.TestCase):
    def test_deep_supervision_is_normalized_and_differentiable(self):
        target = torch.ones(2, 1, 8, 8)
        logit = torch.zeros_like(target, requires_grad=True)
        single = segmentation_loss(logit, target)
        multi = segmentation_loss((logit, logit, logit), target)
        torch.testing.assert_close(single, multi)
        multi.backward()
        self.assertTrue(torch.isfinite(logit.grad).all())

    def test_unet_preserves_dimensions(self):
        model = UNet(base=4).eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 64, 96))
        self.assertEqual(tuple(output.shape), (1, 1, 64, 96))

    def test_gpu_and_protocol_defaults(self):
        cfg = Config()
        self.assertEqual(cfg.gpu, "1")
        self.assertEqual(cfg.size % 14, 0)
        self.assertEqual(cfg.size % 32, 0)
        self.assertEqual(cfg.epochs, 50)
        self.assertEqual(cfg.rle_order, "C")

    def test_persistence_and_concurrent_run_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            atomic_json(path / "status.json", {"state": "completed"})
            self.assertIn("completed", (path / "status.json").read_text())
            with exclusive_lock(path / "gpu.lock"):
                with self.assertRaises(RuntimeError):
                    with exclusive_lock(path / "gpu.lock"):
                        self.fail("Concurrent lock should not succeed")


class OutputIntegrationTests(unittest.TestCase):
    def test_native_size_submission_order_empty_and_full(self):
        class Constant(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, x):
                return torch.full((len(x), 1, 4, 4), self.value)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Test/images").mkdir(parents=True)
            Image.new("RGB", (3, 5)).save(root / "Test/images/000.png")
            Image.new("RGB", (7, 2)).save(root / "Test/images/001.png")
            pd.DataFrame({"id": [1, 0], "target": ["ignored", "ignored"]}).to_csv(
                root / "sample_submission.csv", index=False)
            cfg = replace(Config(), data=str(root), size=32)
            with patch("polyp_benchmark.torch.autocast", side_effect=lambda *a, **k: nullcontext()):
                for value, expected in [(20.0, ["1 14", "1 15"]), (-20.0, ["", ""])]:
                    output = write_submission(Constant(value), cfg, torch.device("cpu"), root)
                    result = pd.read_csv(output, keep_default_na=False)
                    self.assertEqual(result.id.tolist(), [1, 0])
                    self.assertEqual(result.target.tolist(), expected)

    def test_complete_comparison_analysis_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for subdir in ["Dataset/images", "Dataset/masks"]:
                (root / subdir).mkdir(parents=True)
            for name in ["one.png", "two.png"]:
                Image.new("RGB", (30, 20), (100, 100, 100)).save(root / "Dataset/images" / name)
                mask = np.zeros((20, 30), np.uint8)
                mask[5:15, 5:20] = 255
                Image.fromarray(mask).save(root / "Dataset/masks" / name)
            cfg = replace(Config(), data=str(root), output=str(root / "results"))
            self.assertTrue(comparison(cfg, require_all=False).empty)
            with self.assertRaises(RuntimeError):
                comparison(cfg)
            for index, name in enumerate(MODELS):
                dest = Path(cfg.output) / name
                (dest / "val_masks").mkdir(parents=True)
                rows = []
                for image_name in ["one.png", "two.png"]:
                    with Image.open(root / "Dataset/masks" / image_name) as im:
                        mask = np.array(im) > 127
                    Image.fromarray(mask.astype(np.uint8) * 255).save(dest / "val_masks" / image_name)
                    rows.append({"name": image_name, **mask_metrics(mask, mask)})
                pd.DataFrame(rows).to_csv(dest / "val_metrics.csv", index=False)
                pd.DataFrame({"epoch": [1], "val_dice": [1], "train_loss": [0.2]}).to_csv(
                    dest / "history.csv", index=False)
                atomic_json(dest / "result.json", {
                    "model": name, "label": name, "family": "CNN" if index < 3 else "Transformer",
                    "val_dice": 1.0 - index / 100})
            result = analyze(cfg)
            self.assertEqual(len(result), 6)
            self.assertEqual(result.iloc[0].model, "unet")
            for artifact in ["comparison.csv", "paired_bootstrap.csv", "size_analysis.csv",
                             "comparison.png", "qualitative_comparison.png",
                             "analysis_summary.json", "status.json"]:
                self.assertTrue((Path(cfg.output) / artifact).is_file(), artifact)


if __name__ == "__main__":
    unittest.main()
