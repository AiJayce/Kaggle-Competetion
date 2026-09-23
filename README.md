# Xenium Imputation Benchmarking

## (1) Competetion Overview

<img width="741" height="600" alt="1" src="https://github.com/user-attachments/assets/1f5d4ff3-a528-41f2-855b-101570a1a6ae" />

1. Competition Overview 
 - Hosted by Kaggle 
 - Objective: Benchmark methods for imputing missing gene expression in spatial transcriptomics data. 
 - Experimental context: Predict masked gene expression in Xenium spatial transcriptomics data while preserving spatial expression patterns. 

2. Data & Experimental Design 
 - Reference data: scRNA-seq data from the same cell or tissue type, providing gene expression profiles as a reference for learning gene–gene relationships.
 - Training data: Xenium spatial transcriptomics data with known gene expression used to learn relationships between observed and masked genes. 
 - Test data: Xenium spatial transcriptomics data with masked gene expression requiring prediction. 
 - Prediction output: Imputed gene expression values for the masked genes at each spatial location. 

3. Prediction Task 
 - Predict the expression levels of masked genes for each cell while preserving gene–gene expression relationships and spatial expression patterns. 

4. Evaluation 
 - Submissions are evaluated using Pearson Correlation Coefficient (PCC) between predicted and true gene expression. Structural Similarity Index (SSIM) is additionally used to evaluate the similarity of spatial expression patterns. 

5. My Key Concept 
 - GRN-based linear inference is advantageous for predicting masked genes in Xenium data.

## (2) Model Concept

<img width="838" height="683" alt="2" src="https://github.com/user-attachments/assets/fa36fd5c-dc12-464c-aa72-c30df7532bf1" />

On Xenium data, simple regression equations have limitations in capturing nonlinear gene expression relationships. Therefore, a PCC-based gene regulatory network is constructed from scRNA-seq reference data, and module-level regulatory equations are modeled to infer gene expression from a biologically relevant dimensional perspective.

## (3) Build GRN in scRNA-seq

<img width="633" height="673" alt="3" src="https://github.com/user-attachments/assets/eb474a42-ff82-40d3-b11d-c9ce853b204c" />

To establish a ground truth for GRN-based gene expression inference, a PCC-based GRN was constructed from scRNA-seq data, resulting in 22 modules with distinct biological characteristics.

## (4) GRN module Characteristic

<img width="808" height="673" alt="4" src="https://github.com/user-attachments/assets/36503178-cf8e-41d3-b04a-302c65df37a2" />

To validate the applicability of the scRNA-seq-derived GRN to Xenium data, gene overlap between the two datasets was evaluated, along with whether the 22 module scores were distinctly expressed across the slide.

## (5) Gene predicting modeling

<img width="673" height="673" alt="5" src="https://github.com/user-attachments/assets/5e3dfe4b-1f95-48ea-805b-71ba513e41ab" />

To build a regression model for predicting unobserved genes, 20% of genes were masked in the Xenium data, while the remaining 80% were used as training genes. A Ridge regression model was then constructed using module scores, and prediction accuracy was evaluated based on PCC with the ground truth.

<img width="719" height="679" alt="6" src="https://github.com/user-attachments/assets/086a5e88-b948-484f-92f8-ab37638350c1" />

To estimate the accuracy of scRNA-seq-to-Xenium transfer, the expression levels of masked genes predicted by the trained regression model were compared between scRNA-seq and Xenium, and transfer accuracy was assessed using their correlation.

## (6) Spatial Niche-Smoothed Ridge

<img width="760" height="325" alt="7" src="https://github.com/user-attachments/assets/ee0d8557-90cd-4b1c-a26a-4bf980afc988" />

To capture the niche characteristics of spatial transcriptomics, a Niche Neighbor Ensemble Ridge model was developed for fine-tuning. The highest PCC was achieved with an index of 50 and 50 neighbors, resulting in an output that outperformed the first-place score.


## (5) Discussion

### Data and analysis context

The Kaggle Imputation Benchmarking (Xenium Fold 1) competition was used to develop a model for predicting masked gene expression in spatial transcriptomics data. The task provided Xenium data with 105 target genes intentionally masked and 208 observed genes, together with an independent scRNA-seq reference dataset. The Xenium dataset contained 166,363 cells × 313 genes, while the breast cancer scRNA-seq reference dataset contained 30,365 cells × 18,082 genes. Since the Xenium data contained no ground-truth expression values for the 105 target genes, the key challenge was to transfer gene expression relationships learned from scRNA-seq to the Xenium domain.

Rather than predicting each gene independently, gene–gene correlations from the scRNA-seq reference were used to construct a hierarchical Gene Regulatory Network (GRN) structure. Genes were refined into modules using a Winner-Takes-All assignment strategy, and module-level mean expression scores were used as biological representations for each cell. These module representations were combined with spatial neighbor module scores derived from a k-nearest-neighbor graph (k=12), incorporating both gene relationships and spatial context into the model input.

A Ridge regression model was trained to map observed genes to target genes using the scRNA-seq reference. Pseudo-masking and train/validation cell splits were used to estimate prediction performance before applying the model to Xenium data. The trained model was then transferred to the Xenium domain, followed by spatial neighbor smoothing that combined each cell's prediction with predictions from spatially adjacent cells. The smoothing weight was selected through grid search using pseudo-mask validation data. The resulting model was applied to the 105 Xenium target genes without ground-truth labels to generate the final predictions.

### Interpretation

The framework integrated gene-level relationships, GRN-based module representations, and spatial context to predict masked gene expression in Xenium data. The GRN module structure provided a biologically informed representation of gene relationships, while spatial neighbor information incorporated local tissue context into the predictions. Pseudo-masking enabled quantitative validation of the modeling strategy despite the absence of target-gene ground truth in the Xenium domain. This demonstrated an approach for transferring gene expression relationships learned from scRNA-seq to spatial transcriptomics data while incorporating spatial information for prediction refinement.

### Considerations

The predicted Xenium expression depends on the compatibility between the scRNA-seq reference and Xenium data, including differences in tissue composition, sequencing technology, gene coverage, and expression distributions. GRN modules derived from gene–gene correlations represent statistical associations rather than experimentally validated regulatory relationships. In addition, pseudo-masking provides an indirect estimate of performance and may not fully represent prediction accuracy for the truly masked Xenium genes. Independent spatial transcriptomics datasets and directly observed target-gene measurements are required to further evaluate generalization.

### Score Benchmark
| Competition                             | Model Score (PCC) | 1st Place Score (PCC) |
| --------------------------------------- | ----------------: | --------------------: |
| Imputation Benchmarking (Xenium Fold 1) |         **0.575** |             **0.560** |





