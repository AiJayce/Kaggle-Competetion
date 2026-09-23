# Open Problems – Multimodal Single-Cell Integration

## (1) Competetion Overview

<img width="743" height="679" alt="1" src="https://github.com/user-attachments/assets/897588af-58a8-482b-8c7b-5740bb648a01" />

1. Competition Overview
 - Hosted by Kaggle
 - Objective: Predict missing molecular modalities at the single-cell level.
 - Experimental context: Multimodal single-cell data were collected from human CD34+ hematopoietic stem and progenitor cells across multiple differentiation time points.

2. Data & Experimental Design
 - Multiome data: scRNA-seq and scATAC-seq measured simultaneously.
 - CITE-seq data: scRNA-seq and surface protein abundance measured simultaneously.
 - Data were collected across different time points, requiring generalization to later cellular states.

3. Prediction Task
 - Predict the missing molecular modality from the observed modality at the single-cell level.
 - Multiome: Predict gene expression from chromatin accessibility (scATAC → scRNA), using each cell’s regulatory landscape to infer its transcriptional state.
 - CITE-seq: Predict surface protein abundance from gene expression (scRNA → Protein), inferring protein-level cellular states from transcriptomic profiles.

4. Evaluation
 - Submissions are evaluated using Pearson correlation between predicted and ground-truth values.

5. My Key Concept
 - A multimodal diffusion model that jointly models the central dogma (DNA → RNA → Protein) as a unified information framework can improve predictive performance.


## (2) Model Concept

<img width="838" height="680" alt="2" src="https://github.com/user-attachments/assets/706b3233-2a49-4ebe-8688-c409d4fb5aba" />

The central dogma, DNA → RNA → Protein, was modeled as a unified information transformation process. Diffusion noise was interpreted as latent state transitions along diffusion time, enabling the biological information flow between modalities to be modeled through transition matrices. Finally, DNA → RNA and RNA → Protein prediction tasks were then integrated into a single multimodal diffusion framework.

## (3) Preprocess

<img width="701" height="673" alt="3" src="https://github.com/user-attachments/assets/d8b44879-4607-4f61-bf02-c8227665a916" />

Two diffusion models were developed for DNA → RNA and RNA → Protein prediction and then integrated. Both models share same RNA latent space, enabling cross-modal alignment and unified multimodal modeling.

## (4) Diffusion Variance

<img width="743" height="680" alt="4" src="https://github.com/user-attachments/assets/a9f219d1-851f-4312-a9a0-7b286b199134" />

Comparison of the two models showed convergent noise variance across diffusion steps, but substantial differences in overall variance. RNA → Protein showed much higher variance, potentially due to the 23,289 → 128 dimensionality reduction compared with the same 23,289 dimensional ATAC → RNA transition.

## (5) Diffusion – Model performance comparison

<img width="741" height="377" alt="5" src="https://github.com/user-attachments/assets/61e920d9-07d4-4459-a449-747b3320a891" />

The dimensionality mismatch in RNA → Protein led to lower RMSE performance compared with ATAC → RNA. Despite this limitation, the overall model performance remained strong.
This suggests that complementary learning across multiple modalities may have contributed to improved prediction performance.

## (6) Discussion

### Data and analysis context

The Kaggle Open Problems – Multimodal competition was used to develop a multimodal diffusion model for predicting molecular states from single-cell multi-omics data. The competition included RNA expression prediction from scATAC-seq and scRNA-seq Multiome data, as well as Protein expression prediction from scRNA-seq CITE-seq data. Rather than treating these as independent prediction tasks, the framework modeled the flow from DNA to RNA to Protein as a continuous information transformation process using a multimodal diffusion framework.

Two diffusion models were initially designed for the separate prediction tasks. The ATAC–RNA model used promoter and enhancer chromatin accessibility to predict RNA expression, while the RNA–Protein model predicted Protein expression from RNA expression. Each modality was embedded into a latent representation, and the diffusion process was trained through iterative noise addition and denoising. Diffusion time was interpreted as a transition between latent molecular states, allowing information changes across modalities to be modeled.

The RNA representations from CITE-seq and ATAC-Multiome were then aligned within a shared latent space based on their common biological information. This enabled the ATAC–RNA and RNA–Protein diffusion processes to be connected into a continuous multimodal framework representing the DNA–RNA–Protein information flow.

### Interpretation

The multimodal diffusion framework integrated regulatory accessibility, RNA expression, and Protein expression within a shared latent representation. Analysis of the diffusion dynamics showed that noise variance generally converged across diffusion steps in both models, while the RNA–Protein model exhibited higher overall variance. This difference may be related to the substantial dimensionality difference between the high-dimensional RNA and ATAC feature space and the lower-dimensional Protein representation. Despite this difference, the integrated framework achieved competitive prediction performance, suggesting that complementary information across modalities can contribute to multimodal prediction.

### Considerations

The diffusion dynamics and variance differences may be influenced by the dimensionality and statistical characteristics of each modality. The biological interpretation of the shared latent space also depends on how effectively modality-specific representations capture common molecular information. Therefore, further validation using independent multi-omics datasets and additional analyses of latent representations is required to determine whether the learned diffusion transitions accurately reflect biological information flow rather than modality-specific statistical properties.

### Score Benchmark
| Competition                | Model Score (PCC) | 1st Place Score (PCC) |
| -------------------------- | ----------------: | --------------------: |
| Open Problems – Multimodal |         **0.741** |             **0.775** |





