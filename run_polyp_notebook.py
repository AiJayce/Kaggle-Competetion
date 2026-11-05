"""Execute the user's notebook in scrna, persisting outputs after each cell."""

from pathlib import Path
import os
import time
import traceback

import nbformat
from nbclient import NotebookClient

from polyp_benchmark import BASE, Config, atomic_json, exclusive_lock


NOTEBOOK = Path(__file__).with_name("Polyp_Segmentation.ipynb")


def main() -> None:
    cfg = Config()
    out = Path(cfg.output)
    with exclusive_lock(BASE / "notebook.lock"):
        notebook = nbformat.read(NOTEBOOK, as_version=4)

        def persist(**kwargs):
            temporary = NOTEBOOK.with_suffix(".executing.tmp")
            nbformat.write(notebook, temporary)
            temporary.replace(NOTEBOOK)

        def started(cell, cell_index, **kwargs):
            atomic_json(out / "notebook_status.json", {
                "state": "running", "cell_index": cell_index,
                "cell_id": cell.get("id"), "pid": os.getpid(),
                "updated_unix": time.time()})
            print(f"Starting cell {cell_index}: {cell.get('id')}", flush=True)

        client = NotebookClient(
            notebook, kernel_name="scrna", timeout=None, allow_errors=False,
            resources={"metadata": {"path": str(NOTEBOOK.parent)}},
            on_cell_executed=persist, on_cell_start=started,
            record_timing=True)
        try:
            client.execute()
        except Exception as exc:
            persist()
            atomic_json(out / "notebook_status.json", {
                "state": "failed", "error": str(exc), "traceback": traceback.format_exc(),
                "pid": os.getpid(), "updated_unix": time.time()})
            raise
        persist()
        atomic_json(out / "notebook_status.json", {
            "state": "completed", "pid": os.getpid(), "updated_unix": time.time(),
            "submitted_to_kaggle": False})
        print("All six models, analyses and CSV validations completed.", flush=True)


if __name__ == "__main__":
    main()
