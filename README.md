# Polyp Segmentation: Six-Model Benchmark

This branch contains the executed notebook, implementation, tests, dependencies,
and Korean report for benchmarking three CNN and three Transformer segmentation
models. All six model predictions were submitted individually and scored by Kaggle.
These were late submissions, not an official leaderboard placement.

## Files

- [Executed notebook](./Polyp_Segmentation.ipynb): training, evaluation, qualitative analysis, and validation/Public bar plots.
- [Korean model report (PDF)](./Polyp_Segmentation_Model_Report_KO.pdf): how each model learns and infers polyp regions.
- [Benchmark implementation](./polyp_benchmark.py): shared protocol and model factories.
- [Notebook runner](./run_polyp_notebook.py): executes the notebook and persists outputs.
- [Submission companion](./submit_polyp_results.py): validates artifacts and submits each model once.
- [Report generator](./create_polyp_report.py).
- [Dependencies](./requirements-polyp.txt).
- [Benchmark tests](./test_polyp_benchmark.py) and [submission tests](./test_polyp_submission.py).

## Experiment and results

All six models were trained using image/mask pairs: **supervised segmentation**.
DINOv2 uses a self-supervised pretrained encoder, but its segmentation fine-tuning
in this experiment was supervised. U-Net was randomly initialized; the other
models used general pretrained encoders. No polyp-pretrained segmentation weights
were used.

The shared protocol used seed 42, 640 training / 160 validation images, 448 x 448
inputs, BCE plus soft Dice loss, AdamW, effective batch size 8, and up to 50 epochs
with early stopping. Checkpoints were selected using validation mean Dice.
Each model predicted masks for the same 200 test images.

| Model | Validation Dice | Kaggle Public Dice | Kaggle Private Dice |
|---|---:|---:|---:|
| SegFormer | 0.92406 | 0.92330 | 0.90557 |
| Polyp-PVT | 0.91545 | 0.89984 | 0.87426 |
| PraNet | 0.90325 | 0.88121 | 0.87281 |
| HarDNet-MSEG | 0.90119 | 0.88970 | 0.88675 |
| DINOv2 + decoder | 0.79165 | 0.75549 | 0.76854 |
| U-Net | 0.78970 | 0.76284 | 0.77643 |

SegFormer was highest on validation, Public, and Private Dice in this experiment.
This is not a controlled causal comparison of CNNs versus Transformers: model
capacity, decoder architecture, and pretraining differ, and only one image-level
split and seed were evaluated.

[Competition](https://www.kaggle.com/competitions/polyp-segmentation) |
[Imported Kaggle notebook](https://www.kaggle.com/code/kimkiyeont/polyp-segmentation-six-model-benchmark)

## Running and portability

The original files are preserved, including notebook outputs. They reference
the author's local `scrna` environment, workspace paths, GPU 1, and external
dataset/result directories. They are **not a self-contained Kaggle runtime**.
Original images, downloaded upstream repositories, checkpoints, submissions,
and standalone result files live outside the uploaded source folder and are
not included in this branch. Obtain competition data separately and adapt the
paths and device configuration before running elsewhere.

Tests that do not require the training data or external uploads:

```bash
python -m unittest discover -s . -p 'test_polyp_*.py' -v
```

Do not treat historical notebook text saying "no uploads" as the final submission
state: submission was later authorized and completed by the separate companion.
The model report generator also uses external result files and a local Korean font.

## Upstream code permissions

PraNet and Polyp-PVT upstream code is restricted to research/education; separate
permission was confirmed by the user for this experiment's submissions.
HarDNet-MSEG uses Apache-2.0. Those permissions do not automatically grant broader
commercial use or redistribution rights. The original upstream sources are not
vendored in this branch; verify applicable terms before downloading or reuse.

---

# Kaggle Machine Learning Competetion

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/05a51fda-fea8-4685-be29-d16347f9be48" />

This page presents projects related to my participation in Kaggle machine learning competitions.

My personal GitHub profile is available [here](https://github.com/AiJayce).

As Kaggle competitions specifically focused on bioinformatics are relatively limited, opportunities to participate in active competitions were somewhat restricted. For competitions that had already ended, I actively used the Late Submission feature to further develop my modeling skills in relevant domains.

Although projects submitted through Late Submission do not receive an official leaderboard ranking, I objectively evaluated model performance by comparing my results with the scores achieved by top-ranked participants. This provided an opportunity to benchmark the performance of my models against established competition results.

The table below summarizes the competitions I participated in, including the competition name, URL, whether Late Submission was used, and the submission date.

| Competition | Competition URL | My Page | Late Submission | Date |
|---|---|---|:---:|---|
| IMMREP25: TCR Specificity Prediction Challenge | [Kaggle](https://www.kaggle.com/competitions/immrep25) | [GitHub](https://github.com/AiJayce/Kaggle-Competetion/tree/IMMREP25-TCR-Specificity-Prediction-Challenge) | Yes | 2026.08–2026.09 |
| Open Problems – Multimodal Single-Cell Integration | [Kaggle](https://www.kaggle.com/competitions/open-problems-multimodal) | [GitHub](https://github.com/AiJayce/Kaggle-Competetion/tree/Open-Problems-Multimodal-Single-Cell-Integration) | Yes | 2026.09 |
| Open Problems – Single-Cell Perturbations | [Kaggle](https://www.kaggle.com/competitions/open-problems-single-cell-perturbations) | [GitHub](https://github.com/AiJayce/Kaggle-Competetion/tree/Open-Problems-Single-Cell-Perturbations) | Yes | 2026.08 |
| Sartorius – Cell Instance Segmentation | [Kaggle](https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation) | [GitHub](https://github.com/AiJayce/Kaggle-Competetion/tree/Sartorius-Cell-Instance-Segmentation) | Yes | 2026.09 |
| Xenium Imputation Benchmarking | [Kaggle](https://www.kaggle.com/competitions/imputation-benchmarking-xenium-fold-1) | [GitHub](https://github.com/AiJayce/Kaggle-Competetion/tree/Xenium-Imputation-Benchmarking) | Yes | 2026.09 |



Each Github project branch contains the executed code, generated figures, benchmark scores, comparisons with the 1st-place scores, and a summary of the project's significance and key findings.

## Dependency

The dependencies used across all projects are standardized and documented in the following [this file](https://github.com/AiJayce/Kaggle-Competetion/blob/main/DEPENDENCY.csv)
