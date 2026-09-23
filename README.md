# Sartorius-Cell-Instance-Segmentation

## (1) Competetion Overview

<img width="1634" height="1452" alt="0" src="https://github.com/user-attachments/assets/e54d9496-a398-4843-99c6-748b62147985" />

1. Competition Overview
 - Hosted by Kaggle
 - Objective: Detect and segment individual cells from microscopy images.
 - Experimental context: High-resolution microscopy images were collected from different cell types to develop accurate and generalizable cell segmentation models.

2. Data & Experimental Design
 - Microscopy data: Fluorescence microscopy images containing multiple cells with varying shapes, sizes, and spatial distributions.
 - Training data: Images are provided with instance-level segmentation masks identifying each individual cell.
 - The challenge requires the model to distinguish adjacent or overlapping cells as separate cellular instances.

3. Prediction Task
 - Predict pixel-level instance masks to accurately identify and separate individual cells, including neighboring or overlapping cells, from microscopy images.

4. Evaluation
 - Submissions are evaluated using an instance-level mean Average Precision (mAP) metric based on the overlap between predicted and ground-truth cell masks.

5. My Key Concept
 - Data augmentation enhances learning from limited training data, while a mask-based CNN focuses on cell-specific ROIs to improve individual cell segmentation.

## (2) Model Concept

<img width="1898" height="1501" alt="그림1" src="https://github.com/user-attachments/assets/641eab86-94a9-4eac-a0ea-ab3eecc766a1" />

Training on the entire microscopy image may cause empty spaces between cells to be treated as noise, reducing learning efficiency. Therefore, non-ROI regions (empty spaces) are masked before image-based training to improve the efficiency of cell segmentation and annotation.

## (3) Preprocess

<img width="1337" height="1494" alt="그림2" src="https://github.com/user-attachments/assets/691dbbec-2619-4cb0-b0b4-a2aa917faa76" />

For image-based learning, the mask threshold was determined based on the mean Dice score. After ROI selection, image augmentation was applied, followed by embedding generation and CNN-based training.

## (4) Learning rate comparison

<img width="1618" height="1484" alt="그림3" src="https://github.com/user-attachments/assets/a986e276-5239-41b1-90e7-df3cae4dbdc4" />

Data augmentation reduced overfitting, as evidenced by the validation loss, so the augmented model was selected. Overlaying predictions with the ground truth showed that most predictions fell within the GT regions, although the predicted regions tended to be narrower due to the stringent mask threshold.

## (5) Separation confidence comparison

<img width="1633" height="830" alt="그림4" src="https://github.com/user-attachments/assets/fdbbe712-524e-4b61-bede-b29067d6e28c" />

The model achieved a lower score than the competition winner, potentially due to the overly stringent mask threshold. Optimizing the mask threshold using metrics beyond the Dice score may lead to improved performance.

## (6) Discussion

### Data and analysis context

The Kaggle Sartorius Cell Instance Segmentation competition was used to develop a Mask R-CNN model for pixel-level instance segmentation of individual cells in microscopy images. The dataset consisted of 606 microscopy images and 73,585 annotations covering SH-SY5Y, cortical neuron, and astrocyte cells. RLE-based annotations were converted into pixel-level binary masks and bounding boxes to construct the training data. Flip, rotation, and scaling augmentations were applied to improve generalization under limited training data, with identical geometric transformations applied to images and masks to preserve annotation alignment.

Transfer learning was performed using a COCO-pretrained Mask R-CNN with a ResNet-50 FPN backbone. The model was adjusted to detect multiple cell instances in densely packed microscopy images, and the best model was selected based on validation loss. Different mask probability thresholds were evaluated on the validation set to determine the optimal inference condition, after which the predicted masks were converted back to RLE format for submission.

### Interpretation

The model provided an end-to-end computer vision pipeline for biological image analysis, from RLE annotation processing and augmentation to instance detection, segmentation, and final mask generation. Comparison of models with and without augmentation indicated that augmentation helped reduce overfitting and improve generalization to different cell morphologies and spatial distributions.

Analysis of prediction masks relative to ground-truth masks showed that the predicted cell regions tended to be smaller and narrower than the actual cell regions. This suggested that selecting a single mask threshold solely based on Dice score may have limited segmentation performance. A more comprehensive thresholding strategy incorporating cell size, mask confidence, and instance overlap could potentially recover cell boundaries more accurately and improve instance-level segmentation performance.

### Considerations

The relatively lower performance may have been influenced by the use of a single global mask threshold and the difficulty of separating densely packed cell instances with diverse morphologies. Dice-based threshold optimization may not fully reflect the mAP metric used for instance-level evaluation. Further optimization of instance-specific thresholding, post-processing, and mask refinement could improve segmentation quality. Limited GPU memory also required consideration of computational efficiency during data processing and model training.

### Score Benchmark
| Competition                          | Model Score (mAP) | 1st Place Score (mAP) |
| ------------------------------------ | ----------------: | --------------------: |
| Sartorius Cell Instance Segmentation |         **0.286** |             **0.356** |

**Although the score was lower than the 1st-place result, I gained hands-on experience implementing a cell segmentation algorithm.





