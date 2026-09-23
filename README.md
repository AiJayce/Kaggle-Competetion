# Open-Problems-Single-Cell-Perturbations

## (1) Competetion Overview

<img width="743" height="680" alt="그림1" src="https://github.com/user-attachments/assets/a1343796-e68d-41ec-bf77-48d08b2247bc" />

1. Competition Overview
 - Hosted by Kaggle
 - Objective: Predict how small-molecule perturbations alter gene expression across different cell types. 
 - Experimental context: 144 compounds were applied to human PBMCs from three healthy donors, followed by 24-hour single-cell RNA-seq profiling. 

2. Data & Experimental Design
 - Perturbation data: scRNA-seq profiles after drug treatment, with differential expression (DE) calculated for each cell type × compound combination. 
 - Multi-omics data: Baseline 10x Multiome (scRNA-seq + scATAC-seq) data were provided to capture the pre-treatment cellular and chromatin state. 
 - Training data contained all compounds for T/NK cells but only a subset for B/Myeloid cells, creating a perturbation-response generalization problem.

3. Prediction Task
 - Predict the 18,211-gene differential expression profile for unseen B-cell and Myeloid-cell × compound combinations.The task therefore evaluates whether a model can generalize drug-induced transcriptional responses to previously unmeasured cellular contexts.

4. Evaluation
 - Use the Mean Rowwise Root Mean Squared Error to score submissions

5. My Key Concept 
 - Integrating scRNA expression and scATAC peak information through a self-attention block enables the model to better capture cell-type-specific perturbation responses.

## (2) Model Concept

<img width="1001" height="681" alt="그림2" src="https://github.com/user-attachments/assets/73126408-399f-4573-b14d-b025a49cddd2" />

Inferring drug-induced cellular dynamics from RNA expression alone may lack sufficient information to capture nonlinear changes. Therefore, I developed a model that incorporates promoter and enhancer information from ATAC-seq to provide additional dynamic context, enabling more accurate prediction of perturbed gene tokens.

## (3) Preprocess

<img width="492" height="697" alt="그림3" src="https://github.com/user-attachments/assets/bf6f6df2-7570-44dc-84a1-275ba6010eea" />

Generate a gene regulatory attention block using scATAC-seq enhancer peaks, promoter accessibility, and scRNA-seq gene expression.

## (4) Attention block characteristic

<img width="836" height="366" alt="그림4" src="https://github.com/user-attachments/assets/01c1ad3d-95df-427f-9147-3ebc128247ac" />

Generate a gene regulatory attention block using scATAC-seq enhancer peaks, promoter accessibility, and scRNA-seq gene expression.

## (5) Training

<img width="1013" height="687" alt="그림5" src="https://github.com/user-attachments/assets/e6da7b5b-5158-4f72-ab54-58b7f26c048f" />

An MLP was trained within a regulatory attention block to infer nonlinear gene perturbations induced by drug treatment. Residual analysis showed that the residuals for each drug converged toward zero.
Drug response prediction for Myeloid and B cell types after fine-tuning the optimized MLP attention block showed good performance.






