#  Brain Age Prediction using Graph Neural Networks (GATv2)

Explores **brain age prediction** from structural and functional connectomes using **Graph Attention Networks (GATv2)**. The project includes **model training, transfer learning, multimodal architectures, and post-hoc analysis** using SHAP and attention.


##  Datasets

- **ADNI**:  DTI  used for pretraining and best model
- **AD-DECODE**: Full age range (20–90), with gene expression data.
- **ADRC**: Multimodal (DTI + fMRI) and biomarker-rich cohort.



##  Repository Structure

### `ADDECODE Baseline`
 Brain age prediction on the **AD-DECODE** dataset using only **healthy subjects**.
- Full preprocessing pipeline (connectomes + metadata).
- Training GATv2 with node features (FA, MD, Volume) and global features (sex, APOE, BP, graph metrics, PCA genes).
- 7-fold stratified CV, 10 repetitions, early stopping.
- Final model trained on all healthy controls.


### `ADNI Best`
 Training and evaluation on the **ADNI** dataset (70–90 y/o healthy subjects).
- Preprocessing of structural connectomes and regional statistics (FA, Volume).
- GATv2 model training with global features (sex, genotype, graph metrics).


### `Pretrain ADNI-ADDECODE`
 Transfer learning pipeline.
- Uses pretrained ADNI model on all healthy saves it
- Fine-tunes on AD-DECODE data.



### `ADRC Bimodal Best`
 Full **multimodal brain age prediction** using **ADRC** dataset with both **DTI and fMRI** connectomes.
- Dual GATv2 architecture (separate streams for DTI and fMRI).
- Shared node features (FA, Volume), graph metrics (clustering, path length, efficiency), and metadata (sex, genotype).
- Training with 7-fold CV, final model inference on risk subjects (MCI, AD).




### `SHAP-CL`
 **SHAP Contrastive Learning (SHAP-CL)** to improve interpretability.
- Trains the GATv2 model using only healthy control subjects from the AD-DECODE dataset
- Applies the trained model to all subjects (healthy + at-risk) to compute SHAP values for global features.
- Computes cosine similarity between all subject pairs and constructs triplets using the **top-k strategy**
-  Trains a contrastive embedding model (2-layer MLP) using the triplets generated and Optimizes a contrastive loss (NT-Xent) to project SHAP vectors into a latent embedding space
- Uses UMAP for dimensionality reduction and visualizes the 2D projection of SHAP embeddings



### `Analysis Example`
 Example of **post-hoc analysis and plotting**.
- BAG and corrected BAG (cBAG) computation and violin plots comparing BAG/cBAG across clinical risk groups, APOE...
- Performs non-parametric statistical tests (Kruskal-Wallis, Mann-Whitney) to assess group differences in BAG.
- Computes classification performance metrics (AUC, Accuracy, Precision, Recall) based on cBAG thresholds.
- Visualizes global feature importance using SHAP beeswarm plots.
- Computes SHAP values for edge attributes (connectome) using a GNN gradient-based method
- Plot mean SHAP edge importance matrices
- Compares edge-level SHAP values between different age groups
- Creates glass brain visualizations of the most important edges (top-10) across the dataset using Nilearn.
- Regresses cBAG against cognitive test scores.
- Regresses cBAG against regional brain volumes to identify anatomical correlates of accelerated aging.
- Performs unsupervised clustering (e.g., KMeans, UMAP) based on SHAP values to reveal distinct subject profiles.
- Constructs dendrograms based on SHAP vectors and overlays metadata 


### `Subjects count`
 Count and filtering of subjects across datasets.
