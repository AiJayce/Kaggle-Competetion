import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from polyp_benchmark import MODELS, atomic_json, file_hash
from submit_polyp_results import run, validated_submission


class SubmissionTests(unittest.TestCase):
    def make_results(self, root):
        manifest = pd.DataFrame({"id": range(200), "height": [2] * 200, "width": [3] * 200})
        manifest.to_csv(root / "test_manifest.csv", index=False)
        for model in MODELS:
            dest = root / model
            dest.mkdir()
            path = dest / "submission.csv"
            pd.DataFrame({"id": range(200), "target": ["1 6"] * 200}).to_csv(path, index=False)
            atomic_json(dest / "result.json", {
                "submission": str(path), "submission_sha256": file_hash(path),
                "best_epoch": 10, "val_dice": 0.9})

    def test_hash_and_native_dimensions_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_results(root)
            path, _ = validated_submission(root, "unet")
            path.write_text(path.read_text().replace("1 6", "1 7"))
            with self.assertRaisesRegex(ValueError, "hash differs"):
                validated_submission(root, "unet")
            result = json.loads((root / "unet/result.json").read_text())
            result["submission_sha256"] = file_hash(path)
            atomic_json(root / "unet/result.json", result)
            with self.assertRaises(ValueError):
                validated_submission(root, "unet")

    def test_uploads_all_six_once_and_records_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_results(root)

            def execute(command, **kwargs):
                rejected = command[2] == "submit" and "/pranet/" in command[5]
                return subprocess.CompletedProcess(command, 1 if rejected else 0,
                                                    "" if rejected else "ok",
                                                    "deadline exceeded" if rejected else "")

            with patch("submit_polyp_results.shutil.which", return_value="/usr/bin/kaggle"), \
                    patch("submit_polyp_results.subprocess.run", side_effect=execute) as mock:
                run(root, 0)
                self.assertEqual(mock.call_count, 8)
                state = json.loads((root / "kaggle_submission_status.json").read_text())
                self.assertEqual(len(state["attempts"]), 6)
                self.assertEqual(state["state"], "finished_with_submission_failures")
                self.assertEqual(state["attempts"]["pranet"]["state"], "rejected_or_failed")
                run(root, 0)
                self.assertEqual(mock.call_count, 10)  # Only before/after listings, no reuploads.

    def test_interrupted_upload_is_not_automatically_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_results(root)
            atomic_json(root / "kaggle_submission_status.json", {
                "attempts": {"unet": {"state": "submitting"}}})
            with patch("submit_polyp_results.shutil.which", return_value="/usr/bin/kaggle"), \
                    patch("submit_polyp_results.subprocess.run",
                          return_value=subprocess.CompletedProcess([], 0, "", "")) as mock:
                with self.assertRaisesRegex(RuntimeError, "outcome unknown"):
                    run(root, 0)
                self.assertEqual(mock.call_count, 1)

    def test_training_failure_stops_uploads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root / "notebook_status.json", {"state": "failed"})
            with patch("submit_polyp_results.shutil.which", return_value="/usr/bin/kaggle"), \
                    patch("submit_polyp_results.subprocess.run") as mock:
                with self.assertRaisesRegex(RuntimeError, "Training failed"):
                    run(root, 0)
                mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
