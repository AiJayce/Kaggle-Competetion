"""Wait for the validated experiment, then submit each model once to Kaggle.

The user superseded the notebook's original no-upload policy and confirmed
separate license permission for all six models. This companion runs separately
so the already-running training notebook and its immutable protocol stay intact.
Actual upload outcomes live in kaggle_submission_status.json, not in the
notebook's historical submitted_to_kaggle=False training/analysis metadata.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import pandas as pd

from polyp_benchmark import (
    Config, MODELS, atomic_json, exclusive_lock, file_hash, rle_decode, rle_encode,
)


def validated_submission(root: Path, model: str) -> tuple[Path, dict]:
    result = json.loads((root / model / "result.json").read_text())
    path = root / model / "submission.csv"
    if Path(result["submission"]).resolve() != path.resolve():
        raise ValueError(f"{model}: result points to an unexpected submission path")
    if file_hash(path) != result["submission_sha256"]:
        raise ValueError(f"{model}: CSV hash differs from the evaluated artifact")
    manifest = pd.read_csv(root / "test_manifest.csv")
    frame = pd.read_csv(path, keep_default_na=False)
    if list(frame.columns) != ["id", "target"] or len(frame) != 200:
        raise ValueError(f"{model}: invalid CSV schema or row count")
    if frame.id.tolist() != manifest.id.tolist() or not frame.id.is_unique:
        raise ValueError(f"{model}: test IDs differ from the audited manifest")
    for row, item in zip(frame.itertuples(index=False), manifest.itertuples(index=False)):
        mask = rle_decode(row.target, (int(item.height), int(item.width)))
        if rle_encode(mask) != row.target:
            raise ValueError(f"{model}: noncanonical RLE for ID {row.id}")
    return path, result


def run(root: Path, wait_seconds: float) -> None:
    root.mkdir(parents=True, exist_ok=True)
    registry_path = root / "kaggle_submission_status.json"
    with exclusive_lock(root / "kaggle_submission.lock"):
        state = json.loads(registry_path.read_text()) if registry_path.exists() else {
            "competition": "polyp-segmentation",
            "scope": list(MODELS),
            "license_permission": "User confirmed separate permission for all six models.",
            "attempts": {},
        }

        def save(phase: str):
            state.update(state=phase, pid=os.getpid(), updated_unix=time.time())
            atomic_json(registry_path, state)

        try:
            cli = shutil.which("kaggle")
            if cli is None:
                raise RuntimeError("Kaggle CLI is not installed")
            save("waiting_for_all_six_models")
            while not all((root / model / "result.json").exists() for model in MODELS):
                for path in [root / "notebook_status.json"] + [
                        root / model / "status.json" for model in MODELS]:
                    if path.exists():
                        progress = json.loads(path.read_text())
                        if progress.get("state") == "failed":
                            raise RuntimeError(f"Training failed; submission stopped: {path}")
                time.sleep(wait_seconds)
            # Validate every artifact before allowing the first external upload.
            submissions = {model: validated_submission(root, model) for model in MODELS}
            existing = subprocess.run(
                [cli, "competitions", "submissions", "-c", "polyp-segmentation", "--csv"],
                capture_output=True, text=True, timeout=120)
            (root / "kaggle_submissions_before.log").write_text(existing.stdout + existing.stderr)
            if existing.returncode:
                raise RuntimeError("Could not check existing submissions; see kaggle_submissions_before.log")
            for model, (path, result) in submissions.items():
                previous = state["attempts"].get(model)
                if previous is not None:
                    if previous["state"] == "submitting":
                        raise RuntimeError(
                            f"{model}: previous upload outcome unknown; inspect Kaggle before retrying")
                    continue
                message = (
                    f"polyp-six-{model}-{result['submission_sha256'][:12]} "
                    f"seed42 epoch{result['best_epoch']} valDice{result['val_dice']:.5f}")
                if message in existing.stdout:
                    state["attempts"][model] = {
                        "state": "already_present", "message": message,
                        "submission_sha256": result["submission_sha256"]}
                    save("submitting")
                    continue
                attempt = {
                    "state": "submitting", "message": message, "file": str(path),
                    "submission_sha256": result["submission_sha256"],
                    "started_unix": time.time()}
                state["attempts"][model] = attempt
                save("submitting")
                command = [cli, "competitions", "submit", "polyp-segmentation",
                           "-f", str(path), "-m", message]
                # Persist the intent before uploading. An interrupted attempt is not retried.
                completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
                (root / f"kaggle_submit_{model}.log").write_text(
                    completed.stdout + completed.stderr)
                attempt.update(
                    state="accepted_by_cli" if completed.returncode == 0 else "rejected_or_failed",
                    returncode=completed.returncode, finished_unix=time.time(),
                    stdout=completed.stdout, stderr=completed.stderr)
                save("submitting")
                print(f"{model}: {attempt['state']}", flush=True)
            listing = subprocess.run(
                [cli, "competitions", "submissions", "-c", "polyp-segmentation", "--csv"],
                capture_output=True, text=True, timeout=120)
            (root / "kaggle_submissions_after.log").write_text(listing.stdout + listing.stderr)
            state["listing_returncode"] = listing.returncode
            # CLI acceptance is not a completed score; preserve the server's status separately.
            statuses = [attempt["state"] for attempt in state["attempts"].values()]
            save("uploads_attempted" if all(
                s in ("accepted_by_cli", "already_present") for s in statuses)
                else "finished_with_submission_failures")
        except Exception as exc:
            state["error"] = str(exc)
            state["traceback"] = traceback.format_exc()
            save("failed")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(Config().output))
    args = parser.parse_args()
    run(args.output, wait_seconds=20)
