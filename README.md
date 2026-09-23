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

## (5) Discussion

### Data and analysis context

The Kaggle Open Problems – Single-cell Perturbations competition was used to develop a deep learning model for predicting drug-induced gene expression changes at the single-cell level. The training data consisted of pre-treatment scRNA-seq and scATAC-seq single-cell multiome data from CD4+ T cells, CD8+ T cells, Regulatory T cells, NK cells, Myeloid cells, and B cells, together with post-treatment DEG data from 126 drugs in CD4+ T cells, CD8+ T cells, Regulatory T cells, and NK cells. Since post-treatment labels were not provided for Myeloid cells and B cells, drug responses for these cell types were predicted by learning perturbation patterns from the observed cell types.

A gene-token-based Self-Attention architecture was implemented in PyTorch to integrate gene expression and regulatory accessibility information. Gene expression and chromatin accessibility were embedded into a regulatory representation, followed by Self-Attention to capture gene-level interactions and perturbation context. An MLP was then trained using post-treatment DEG profiles as labels. Dropout and Early Stopping were applied to control overfitting. The trained model was subsequently fine-tuned on Myeloid cells and B cells to transfer the learned representations to cell types without direct perturbation labels.

### Interpretation

The model integrated gene expression and regulatory information from single-cell multiome data to predict drug-induced transcriptional responses. The Self-Attention architecture captured gene-level interactions and regulatory context under perturbation, while fine-tuning enabled the learned representations from observed cell types to be transferred to Myeloid cells and B cells. This framework enabled the prediction of perturbation responses in cell types without direct post-treatment observations.

### Considerations

The predicted drug responses depend on the quality and distribution of the single-cell multiome data, drug-specific perturbation characteristics, and the biological similarity between the training and target cell types. Since Myeloid cells and B cells lacked direct post-treatment labels, their predicted responses represent transferred model estimates rather than directly observed perturbation effects. Independent datasets and experimental validation are required to assess the generalization and biological relevance of the predictions.

### Score Benchmark
| Competition                               | Model Score (MRRMSE) | 1st Place Score (MRRMSE) |
| ----------------------------------------- | -------------------: | -----------------------: |
| Open Problems – Single-cell Perturbations |            **0.826** |                **0.729** |




