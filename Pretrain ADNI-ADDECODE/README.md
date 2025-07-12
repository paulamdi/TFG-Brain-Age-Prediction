
## 0_PearsonCorrelation.py

This script computes Pearson correlations of brain connectomes across age groups using data from both AD-DECODE and ADNI datasets.

- Loads and preprocesses connectome matrices and metadata for both datasets.
- Filters to include only healthy control (CN) subjects.
- Organizes subjects into predefined age bins (20–30, 30–40, ..., 80–90).
- Computes average connectome vectors per age group.
- Calculates intra-group and cross-dataset Pearson correlations:
  - Within AD-DECODE.
  - Within ADNI.
  - Between AD-DECODE and ADNI.
  - Subject-to-subject across datasets.

## 1_GAT adni.py - ADNI Pretraining 

This script implements **pretraining on the ADNI dataset** using a Graph Attention Network (GATv2) model.  
The architecture and preprocessing steps were carefully adapted to **maximize compatibility with the AD-DECODE dataset**, facilitating better transfer learning.

Although more graph metrics (e.g., efficiency) were available in ADNI, this setup intentionally **excludes features not used in AD-DECODE**.

The ADNI model mirrors the AD-DECODE model, but **without**:
 - **Mean Diffusivity (MD)**  
 - **Blood Pressure** (Systolic, Diastolic)  
 - **Transcriptomic PCA components**

###  Features used in ADNI pretraining:

- **Node features**:
  - Fractional Anisotropy (FA)
  - Regional Volume
  - Node-wise Clustering Coefficient

- **Global features**:
  - Sex (one-hot)
  - APOE genotype (label-encoded)
  - Global Clustering Coefficient
  - Path Length

###  Preprocessing:

- Connectomes:
  - Thresholded at the top 5% of connections
  - Log(x + 1) transformation applied
- Node features:
  - Z-score normalization (node-wise)
- Global features:
  - Encoded and concatenated with graph metrics

###  Architecture:

- 4-layer GATv2 with residual connections  
- Batch normalization and dropout  
- Global feature MLP with fusion  
- Final MLP head for brain age regression

###  Training Strategy:

- 7-fold stratified cross-validation by age  
- 10 repetitions per fold  
- Early stopping (patience = 40)  
- Loss: SmoothL1Loss  
- Optimizer: AdamW  
- Scheduler: StepLR  

After CV, the model was retrained on **all healthy ADNI subjects** and saved 



