# Polyp Segmentation Benchmark

## (1) Introduction

![Example colonoscopy image containing a polyp](images/figure0.png)

1. Competition Overview
 - Hosted by Kaggle
 - Objective: Benchmark deep learning methods for colorectal polyp segmentation.
 - Experimental context: Pixel-level polyp segmentation in colonoscopy images.

2. Data & Experimental Design
Training data: Colonoscopy images with ground-truth segmentation masks.
Test data: Colonoscopy images without segmentation masks.
Annotation: Pixel-level masks for polyp regions and background.
Prediction output: Pixel-wise segmentation masks.

3. Prediction Task
 - Identify polyp regions while accurately capturing boundaries, shapes, and spatial structures.

4. Evaluation
 - Mean Dice coefficient between predicted and ground-truth masks.
 - Higher Dice scores indicate better segmentation performance.
 - Predictions submitted in Run-Length Encoding (RLE) format.

5. My Key Concept
- Benchmark multiple CNN and Vision Transformer (ViT) architectures to compare their suitability for polyp segmentation.

Accurate polyp segmentation in colonoscopy images can help identify the location and extent of lesions. This project trains and compares six deep-learning models that predict a pixel-level polyp mask from an endoscopic image.

The benchmark includes three CNN-based models—U-Net, PraNet, and HarDNet-MSEG—and three Transformer-based models—DINOv2 with a segmentation decoder, SegFormer, and Polyp-PVT. All models were evaluated using the same training/validation split and a consistent evaluation protocol. Validation metrics and Kaggle Public scores were compared to assess how well validation performance reflected submission performance.

## (2) Training Data and Procedure

![Training data and segmentation workflow](images/figure1.png)

The training set consists of colonoscopy images paired with binary ground-truth polyp masks. Each model learns to predict which pixels belong to a polyp.

All models used the same preprocessing and training protocol:

| Setting | Description |
|---|---|
| Data split | 640 training images and 160 validation images |
| Input | RGB images resized to 448 × 448 |
| Normalization | ImageNet mean and standard deviation |
| Mask processing | Nearest-neighbor resizing followed by binary thresholding |
| Augmentation | 90-degree rotations, horizontal and vertical flips, and brightness/contrast adjustments; training set only |
| Loss | Binary Cross-Entropy + Soft Dice Loss |
| Optimizer | AdamW |
| Learning-rate schedule | Cosine scheduler |
| Training budget | Up to 50 epochs, with early stopping |
| Model selection | Checkpoint with the highest validation mean Dice |

The same split and shared training settings were used for all six models. Since architectures and pretraining differ, this benchmark compares the models under a common protocol rather than isolating architecture as the only variable.

## (3) Benchmark Model Categories

![CNN and Transformer model categories](images/figure2.png)

The benchmark compares two broad model groups:

- **CNN-based:** U-Net, PraNet, and HarDNet-MSEG
- **Transformer-based:** DINOv2 + decoder, SegFormer, and Polyp-PVT

These models were selected to compare different feature-extraction and segmentation approaches on the same task. DINOv2 is a pretrained encoder paired with a segmentation decoder; it is not a segmentation architecture by itself.

## (4) Training Progress

![Training progress and validation performance](images/figure3.png)

The training curves show how validation Dice and training loss changed across epochs. In general, training loss decreased as learning progressed, while validation Dice increased and eventually stabilized.

The curves provide an early indication of which models may perform well, but training loss alone does not establish generalization performance. Final model comparisons therefore use validation metrics and Kaggle Public scores. In this experiment, SegFormer reached the strongest overall validation performance, while Polyp-PVT and PraNet also achieved high validation Dice.

## (5) Qualitative Prediction Comparison

![Ground-truth masks and model predictions](images/figure4.png)

The figure compares the ground-truth mask with predictions from each model on selected validation images. Overlay colors indicate whether each pixel was correctly or incorrectly classified:

- **Green:** True Positive — predicted as polyp and present in the ground truth
- **Red:** False Positive — predicted as polyp but absent from the ground truth
- **Blue:** False Negative — present in the ground truth but missed by the model

A larger green region and smaller red and blue regions indicate closer agreement with the ground truth. These examples also illustrate model-specific over-segmentation and missed regions. The overlays are qualitative examples and should be interpreted alongside the numerical validation metrics.

## (6) Validation Benchmark

![Validation benchmark across segmentation metrics](images/figure5.png)

The 800 paired training images were split into 640 training images and 160 validation images using a fixed random seed. The validation set was not used for gradient updates; it was used for model selection and performance evaluation.

| Data split | Number of images | Purpose |
|---|---:|---|
| Training set | 640 | Model training |
| Validation set | 160 | Model selection and evaluation |
| Kaggle test set | 200 | Submission scoring |

Validation results are shown below. Mean Dice is averaged across images, while Global Dice is calculated by aggregating pixel-level true positives, false positives, and false negatives across the dataset. The two metrics may therefore produce different rankings.

| Model | Mean Dice | Mean IoU | Precision | Recall | Boundary F1 | Global Dice |
|---|---:|---:|---:|---:|---:|---:|
| U-Net | 0.7897 | 0.6898 | 0.8119 | 0.8370 | 0.4816 | 0.7833 |
| PraNet | 0.9033 | 0.8413 | 0.9160 | **0.9167** | 0.7087 | 0.8963 |
| HarDNet-MSEG | 0.9012 | 0.8393 | 0.9236 | 0.9092 | 0.7102 | 0.8978 |
| DINOv2 + decoder | 0.7917 | 0.6928 | 0.8168 | 0.8283 | 0.4663 | 0.8112 |
| SegFormer | **0.9241** | **0.8755** | 0.9361 | 0.9317 | **0.7543** | **0.9268** |
| Polyp-PVT | 0.9154 | 0.8611 | **0.9380** | 0.9145 | 0.7290 | 0.9208 |

SegFormer achieved the highest Mean Dice, Mean IoU, Boundary F1, and Global Dice. Polyp-PVT had the highest precision, while PraNet had the highest recall. Overall, SegFormer produced the strongest validation results across the evaluated metrics.

## (7) Final Score Benchmark

![Validation Dice and Kaggle Public Dice comparison](images/figure6.png)

The table below ranks the six models by their Kaggle Public Dice score.

| Rank | Model | Validation Dice | Kaggle Public Dice |
|---:|---|---:|---:|
| 1 | SegFormer | 0.9241 | **0.9233** |
| 2 | Polyp-PVT | 0.9154 | **0.8998** |
| 3 | HarDNet-MSEG | 0.9012 | **0.8897** |
| 4 | PraNet | 0.9033 | **0.8812** |
| 5 | U-Net | 0.7897 | **0.7628** |
| 6 | DINOv2 + decoder | 0.7917 | **0.7555** |

SegFormer achieved the highest Kaggle Public score among the six evaluated models, followed by Polyp-PVT and HarDNet-MSEG. The overall ordering was broadly consistent with the validation results, although validation performance does not perfectly predict a model's score on unseen test data. These scores compare only the six models in this benchmark and do not represent the full competition leaderboard.

## Discussion

Under the shared data split and training protocol, SegFormer achieved the strongest overall validation and Kaggle Public results. Polyp-PVT performed best in precision, while PraNet achieved the highest recall. U-Net and DINOv2 + decoder had lower validation and Public scores in this experiment.

These findings are specific to the dataset split, preprocessing, and training settings used here; they should not be assumed to hold for other datasets or experimental setups. Since metrics such as Mean Dice, Global Dice, precision, and recall capture different aspects of segmentation quality, model selection should consider the intended application as well as the overall score.

For implementation and experiment details, see the [training notebook](./Polyp_Segmentation.ipynb) and [benchmark script](./polyp_benchmark.py).
