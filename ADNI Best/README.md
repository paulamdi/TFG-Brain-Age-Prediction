# Brain Age Prediction from ADNI Connectomes using GATv2 (BEST ADNI not the one used for pretraining)

This repository implements a full pipeline for brain age prediction using **structural connectomes** from the **ADNI** dataset. The model leverages **Graph Attention Networks (GATv2)** with multimodal node features (FA, Volume) and global metadata (sex, genotype, graph metrics). The project includes preprocessing, training with stratified cross-validation, and final pretraining for transfer learning.


### 1. Data Loading and Preprocessing
- Load **ADNI connectomes** (`onn_plain.csv`) from individual subject folders.
- Match connectomes with metadata (`idaSearch_3_19_2025FINAL.xlsx`) using subject ID and visit code (`y0`, `y4`).
- Filter to include **only healthy control (CN)** subjects.
- Extract **node features** per region:
  - FA (Fractional Anisotropy)
  - Volume
- Normalize node features **node-wise** across subjects.
- Apply log(x+1) transform and **95th percentile thresholding** (keep 95%) to connectomes.

### 2. Graph Construction
Each subject is converted into a `torch_geometric.data.Data` graph:
- `x`: [84, 2] node features (FA, Volume)
- `edge_index`, `edge_attr`: edges from upper triangular matrix
- `y`: chronological age
- `global_features`: concatenation of
  - Encoded sex and APOE genotype
  - 4 global graph metrics:
    - Clustering Coefficient
    - Characteristic Path Length
    - Global Efficiency
    - Local Efficiency

##  Model Architecture (BrainAgeGATv2_ADNI)

A 4-layer GATv2 model with residual connections and batch normalization:

- **Node stream**:  
  - Linear(2 → 64) → 4 × GATv2Conv (edge-aware) → Global Mean Pool
- **Global stream** (multi-head MLP):
  - `meta_head`: sex + genotype → 16-dim
  - `graph_head`: 4 graph metrics → 32-dim
- **Fusion**: Concatenate pooled node and global features (128 + 48)
- Final regression head outputs predicted brain age.


##  Model Training and Evaluation

### Cross-Validation
- **7-fold stratified CV**, repeated 10 times (70 total runs)
- Stratification by age bins
- **SmoothL1Loss** (β = 1), optimizer: AdamW
- **Early stopping** with patience = 40
- Learning rate scheduler: StepLR (step=20, gamma=0.5)

### Evaluation Metrics
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)
- RMSE (Root Mean Squared Error)

Results are reported as **mean ± std** across all folds and repetitions.

### Visualizations
- Learning curves (mean ± std)
- True vs predicted ages with performance metrics
- Fold-wise metrics printed to console


##  Final Pretraining on All Data

Once cross-validation confirms robustness, the model is retrained on **all healthy ADNI subjects**:
- Epochs = 150 (based on early stopping behavior)
- Final weights saved 



