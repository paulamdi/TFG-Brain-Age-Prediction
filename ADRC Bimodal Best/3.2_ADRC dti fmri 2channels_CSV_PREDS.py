#ADRC DTI fmri 2 channels bimodal 
#with rmse
#With biomarkers


import os
import pandas as pd
import numpy as np
import random
import torch



import os

output_dir = "bimodal_training_eval_plots_save"
os.makedirs(output_dir, exist_ok=True)




def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


#LOAD CONNECTOMES
print ("LOAD CONNECTOMES DTI")

base_path_adrc_dti = "/home/bas/Desktop/MyData/ADRC/data/ADRC_connectome_bank/connectome/DTI/plain"
adrc_dti_connectomes = {}

for filename in os.listdir(base_path_adrc_dti):
    # Skip hidden system files or ._macOS files
    if not filename.endswith("_conn_plain.csv") or filename.startswith("._"):
        continue

    subject_id = filename.replace("_conn_plain.csv", "")  # e.g., ADRC0001
    file_path = os.path.join(base_path_adrc_dti, filename)
    try:
        matrix = pd.read_csv(file_path, header=None)
        adrc_dti_connectomes[subject_id] = matrix
    except Exception as e:
        print(f"Error loading {filename}: {e}")

print(f" Total ADRC DTI connectomes loaded: {len(adrc_dti_connectomes)}")
print(" Example subject IDs:", list(adrc_dti_connectomes.keys())[:5])
print()





print("LOAD CONNECTOMES fMRI")

base_path_adrc_fmri = "/home/bas/Desktop/MyData/ADRC/data/ADRC_connectome_bank/connectome/fMRI/corr"
adrc_fmri_connectomes = {}

for filename in os.listdir(base_path_adrc_fmri):
    if not filename.startswith("func_connectome_corr_ADRC") or not filename.endswith(".csv"):
        continue

    subject_id = filename.replace("func_connectome_corr_", "").replace(".csv", "")  # e.g., ADRC0001
    file_path = os.path.join(base_path_adrc_fmri, filename)
    try:
        matrix = pd.read_csv(file_path, header=None)
        adrc_fmri_connectomes[subject_id] = matrix
    except Exception as e:
        print(f"Error loading {filename}: {e}")

print(f" Total ADRC fMRI connectomes loaded: {len(adrc_fmri_connectomes)}")
print(" Example subject IDs:", list(adrc_fmri_connectomes.keys())[:5])
print()



# Intersect subjects that have both DTI and fMRI
matched_dti_fmri = sorted(set(adrc_dti_connectomes.keys()) & set(adrc_fmri_connectomes.keys()))

print(f" Matched subjects with both DTI and fMRI: {len(matched_dti_fmri)}")
print(" Example matched subjects:", matched_dti_fmri[:5])
print()



#LOAD METADATA
print("LOAD METADATA ")


# === Step 1: Define the metadata file path ===
metadata_path_adrc = "/home/bas/Desktop/MyData/ADRC/data/ADRC_connectome_bank/metadata/alex-badea_2024-06-14 (copy).xlsx"

# === Step 2: Load the Excel file into a DataFrame ===
# This will read the first sheet by default
df_adrc_meta= pd.read_excel(metadata_path_adrc)

# === Step 3: Display metadata summary ===
print(" ADRC metadata loaded successfully.")
print(" Metadata shape:", df_adrc_meta.shape)
print(" Preview of first rows:")
print(df_adrc_meta.head())
print()






#MATCH CONNECTMES (has both) AND METADATA
print("=== MATCHING CONNECTOMES AND METADATA (DTI + fMRI only) ===")

# List of subjects that have both DTI and fMRI connectomes
matched_dti_fmri = sorted(set(adrc_dti_connectomes.keys()) & set(adrc_fmri_connectomes.keys()))

# Get subject IDs present in the metadata
metadata_subject_ids = set(df_adrc_meta["PTID"])

# Find subjects that have DTI, fMRI, and metadata
matched_ids = sorted(set(matched_dti_fmri) & metadata_subject_ids)

# Filter metadata to include only matched subjects
df_matched_adrc = df_adrc_meta[df_adrc_meta["PTID"].isin(matched_ids)].copy()

# Filter DTI connectomes to matched subjects
adrc_dti_connectomes_matched = {
    sid: adrc_dti_connectomes[sid]
    for sid in matched_ids
}

# Filter fMRI connectomes to matched subjects
adrc_fmri_connectomes_matched = {
    sid: adrc_fmri_connectomes[sid]
    for sid in matched_ids
}

# Print summary of matching
print(f"Subjects with DTI + fMRI: {len(matched_dti_fmri)}")
print(f"Subjects with metadata: {len(metadata_subject_ids)}")
print(f"Matched subjects (DTI + fMRI + metadata): {len(matched_ids)}")
print(f"- Rows in matched metadata: {df_matched_adrc.shape[0]}")
print(f"- DTI connectomes matched: {len(adrc_dti_connectomes_matched)}")
print(f"- fMRI connectomes matched: {len(adrc_fmri_connectomes_matched)}")
print("Example matched subject IDs:", matched_ids[:5])
print()





##########################################
# FILTER TO NON-DEMENTED SUBJECTS (DEMENTED ≠ 1)
##########################################

# Step 1: Keep only non-demented rows from the combined metadata (DTI + fMRI + metadata)
df_healthy_adrc = df_matched_adrc[
    (df_matched_adrc["DEMENTED"] != 1) | (df_matched_adrc["DEMENTED"].isna())
].copy()

# Step 2: Get IDs of non-demented subjects
healthy_subject_ids = df_healthy_adrc["PTID"].tolist()

# Step 3: Filter connectomes (DTI and fMRI) to include only non-demented subjects
adrc_dti_connectomes_healthy = {
    sid: adrc_dti_connectomes_matched[sid]
    for sid in healthy_subject_ids if sid in adrc_dti_connectomes_matched
}

adrc_fmri_connectomes_healthy = {
    sid: adrc_fmri_connectomes_matched[sid]
    for sid in healthy_subject_ids if sid in adrc_fmri_connectomes_matched
}

# Step 4: Print summary
print(f" Number of non-demented subjects: {len(healthy_subject_ids)}")
print(f" DTI connectomes available (non-demented): {len(adrc_dti_connectomes_healthy)}")
print(f" fMRI connectomes available (non-demented): {len(adrc_fmri_connectomes_healthy)}")
print()




# df_healthy_adrc_dti -> dataframe healthy

# adrc_dti_connectomes_healthy -> dic connectomes helathy
# adrc_fmri_connectomes_healthy






# FA, VOLUME





##### FA

import torch

# === Get valid subjects: healthy ADRC subjects with connectome and metadata
valid_subjects = set(df_healthy_adrc["PTID"])



# === Paths to stats files ===
fa_path = "/home/bas/Desktop/MyData/ADRC/data/ADRC_connectome_bank/ADRC_Regional_Stats/studywide_stats_for_fa.txt"



df_fa = pd.read_csv(fa_path, sep="\t")

# === Remove ROI 0
df_fa = df_fa[df_fa["ROI"] != 0].reset_index(drop=True)

# === Select only columns for valid subjects
subject_cols_fa = [col for col in df_fa.columns if col in valid_subjects]
df_fa_transposed = df_fa[subject_cols_fa].transpose()

# === Set subject ID and convert to float
df_fa_transposed.index.name = "subject_id"
df_fa_transposed = df_fa_transposed.astype(float)






#####VOLUME

# === Load Volume data ===
vol_path = "/home/bas/Desktop/MyData/ADRC/data/ADRC_connectome_bank/ADRC_Regional_Stats/studywide_stats_for_volume.txt"
df_vol = pd.read_csv(vol_path, sep="\t")



# === Remove ROI 0 row, as it is not a brain region of interest
df_vol = df_vol[df_vol["ROI"] != 0].reset_index(drop=True)

# === Select only columns corresponding to valid subjects
subject_cols_vol = [col for col in df_vol.columns if col in valid_subjects]
df_vol_transposed = df_vol[subject_cols_vol].transpose()

# === Set subject IDs as index and convert to float
df_vol_transposed.index.name = "subject_id"
df_vol_transposed = df_vol_transposed.astype(float)





# === Combine FA and Volume into [84, 2] tensors per subject ===
multimodal_features_dict = {}

for subj in df_fa_transposed.index:
    if subj in df_vol_transposed.index:
        fa = torch.tensor(df_fa_transposed.loc[subj].values, dtype=torch.float32)
        vol = torch.tensor(df_vol_transposed.loc[subj].values, dtype=torch.float32)

        if fa.shape[0] == 84 and vol.shape[0] == 84:  # safety check
            stacked = torch.stack([fa, vol], dim=1)  # shape: [84, 2]
            multimodal_features_dict[subj] = stacked
        else:
            print(f" Subject {subj} has unexpected feature shape: FA={fa.shape}, VOL={vol.shape}")

print(f" Subjects with valid multimodal node features: {len(multimodal_features_dict)}")

#check
# Print just the first subject key
first_subj = list(multimodal_features_dict.keys())[0]
print("First subject ID:", first_subj)

# Print shape and a snippet of its tensor
print("Feature shape:", multimodal_features_dict[first_subj].shape)
print("Feature preview:\n", multimodal_features_dict[first_subj][:5])  # print first 5 nodes






# === Normalize node features (across subjects, per node+modality) ===
def normalize_multimodal_nodewise(feature_dict):
    all_features = torch.stack(list(feature_dict.values()))  # shape: [N_subjects, 84, 2]
    means = all_features.mean(dim=0)  # shape: [84, 2]
    stds = all_features.std(dim=0) + 1e-8  # shape: [84, 2] to avoid div by zero
    return {subj: (features - means) / stds for subj, features in feature_dict.items()}

normalized_node_features_dict = normalize_multimodal_nodewise(multimodal_features_dict)

print(f" Node features normalized. Example shape: {list(normalized_node_features_dict.values())[0].shape}")











# === Matrix to graph function (ADRC) ===
def matrix_to_graph(matrix, device, subject_id, node_features_dict):
    indices = np.triu_indices(84, k=1)  # upper triangle (excluding diagonal)
    edge_index = torch.tensor(np.vstack(indices), dtype=torch.long, device=device)
    edge_attr = torch.tensor(matrix.values[indices], dtype=torch.float32, device=device)

    # Get node features for this subject
    node_feats = node_features_dict[subject_id]  # shape: [84, 2]
    node_features = node_feats.to(device)

    return edge_index, edge_attr, node_features






# Print dti and fmri connectomes before log and th

import matplotlib.pyplot as plt
import seaborn as sns

# === Select subject to visualize ===
subject_id = list(adrc_dti_connectomes_healthy.keys())[0]

# === Retrieve DTI and fMRI matrices (raw) ===
dti_matrix = adrc_dti_connectomes_healthy[subject_id]
fmri_matrix = adrc_fmri_connectomes_healthy[subject_id]

# === Retrieve age from metadata ===
age = df_healthy_adrc[df_healthy_adrc["PTID"] == subject_id]["SUBJECT_AGE_SCREEN"].values[0]

# === Plot side-by-side heatmaps (DTI + fMRI raw) ===
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.heatmap(dti_matrix, cmap="viridis", square=True, cbar=True, xticklabels=False, yticklabels=False)
plt.title(f"DTI (Raw) - {subject_id} (Age {age:.1f})")

plt.subplot(1, 2, 2)
sns.heatmap(fmri_matrix, cmap="viridis", square=True, cbar=True, xticklabels=False, yticklabels=False)
plt.title(f"fMRI (Raw) - {subject_id} (Age {age:.1f})")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "heatmaps_before_log_and_th.png"), dpi=300)
plt.close()



# === Print numeric values of DTI ===
print(f"\nNumeric DTI Matrix for subject {subject_id} (Age {age:.1f}):")
print(dti_matrix.to_numpy())

# === Print numeric values of fMRI ===
print(f"\nNumeric fMRI Matrix for subject {subject_id} (Age {age:.1f}):")
print(fmri_matrix.to_numpy())






#LOG AND THRESHOLD


# === Thresholding function ===
def threshold_connectome(matrix, percentile=95):
    matrix_np = matrix.to_numpy()
    mask = ~np.eye(matrix_np.shape[0], dtype=bool)
    values = matrix_np[mask]
    threshold_value = np.percentile(values, 100 - percentile)
    thresholded = np.where(matrix_np >= threshold_value, matrix_np, 0)
    symmetrized = np.maximum(thresholded, thresholded.T)
    return pd.DataFrame(symmetrized, index=matrix.index, columns=matrix.columns)









#DTI 

# === Output dictionary ===
log_thresholded_connectomes_adrc_dti_healthy = {}

# === Track issues ===
invalid_shape_ids = []
failed_processing_ids = []

# === Apply to healthy subjects ===
for subject_id, matrix in adrc_dti_connectomes_healthy.items():
    try:
        # Check shape (e.g., 84x84)
        if matrix.shape != (84, 84):
            invalid_shape_ids.append(subject_id)
            continue

        # Apply threshold and log1p
        thresholded = threshold_connectome(matrix, percentile=95)
        log_matrix = np.log1p(thresholded)
        log_thresholded_connectomes_adrc_dti_healthy[subject_id] = log_matrix

    except Exception as e:
        failed_processing_ids.append(subject_id)
        print(f" Error processing subject {subject_id}: {e}")

# === Summary ===

print(f" Total healthy DTI connectomes processed (threshold + log): {len(log_thresholded_connectomes_adrc_dti_healthy)}")


### log_thresholded_connectomes_adrc_dti_healthy = {}  -> dic Only healthy th and log connectomes dti adrc




#fMRI (NO LOG NEGATIVES...)

# === Output dictionary ===
log_thresholded_connectomes_adrc_fmri_healthy = {}

# === Track issues ===
invalid_shape_ids_fmri = []
failed_processing_ids_fmri = []

# === Apply to healthy subjects ===
for subject_id, matrix in adrc_fmri_connectomes_healthy.items():
    try:
        if matrix.shape != (84, 84):
            invalid_shape_ids_fmri.append(subject_id)
            continue

        # Threshold + log transform
        thresholded = threshold_connectome(matrix, percentile=95)  # or maybe 90 for fMRI
        log_matrix = np.log1p(thresholded)
        log_thresholded_connectomes_adrc_fmri_healthy[subject_id] = log_matrix

    except Exception as e:
        failed_processing_ids_fmri.append(subject_id)
        print(f" Error processing subject {subject_id}: {e}")

print(f" Total healthy fMRI connectomes processed (threshold + log): {len(log_thresholded_connectomes_adrc_fmri_healthy)}")
print()




#PRINT AFTER LOG AND TH

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.heatmap(log_thresholded_connectomes_adrc_dti_healthy[subject_id],
            cmap="viridis", square=True, cbar=True, xticklabels=False, yticklabels=False)
plt.title(f"DTI (Log+TH) - {subject_id} (Age {age:.1f})")

plt.subplot(1, 2, 2)
sns.heatmap(log_thresholded_connectomes_adrc_fmri_healthy[subject_id],
            cmap="viridis", square=True, cbar=True, xticklabels=False, yticklabels=False)
plt.title(f"fMRI (Log+TH) - {subject_id} (Age {age:.1f})")

plt.tight_layout()

plt.savefig(os.path.join(output_dir, "heatmaps_after_log_th.png"), dpi=300)
plt.close()



#Max and min values
import numpy as np

def get_upper_values(matrix_dict):
    values = []
    for mat in matrix_dict.values():
        arr = mat.values if isinstance(mat, pd.DataFrame) else mat
        upper = arr[np.triu_indices(arr.shape[0], k=1)]  # upper triangle without diagonal
        values.append(upper.flatten())
    return np.concatenate(values)

# === DTI RAW ===
dti_raw_vals = get_upper_values(adrc_dti_connectomes_healthy)
print("DTI RAW")
print(f"Min: {dti_raw_vals.min():.2f} | Max: {dti_raw_vals.max():.2f} | Mean: {dti_raw_vals.mean():.2f} | Std: {dti_raw_vals.std():.2f}")
print()

# === fMRI RAW ===
fmri_raw_vals = get_upper_values(adrc_fmri_connectomes_healthy)
print("fMRI RAW")
print(f"Min: {fmri_raw_vals.min():.2f} | Max: {fmri_raw_vals.max():.2f} | Mean: {fmri_raw_vals.mean():.2f} | Std: {fmri_raw_vals.std():.2f}")
print()

# === DTI LOG+TH ===
dti_log_vals = get_upper_values(log_thresholded_connectomes_adrc_dti_healthy)
print("DTI (Log+TH)")
print(f"Min: {dti_log_vals.min():.2f} | Max: {dti_log_vals.max():.2f} | Mean: {dti_log_vals.mean():.2f} | Std: {dti_log_vals.std():.2f}")
print()

# === fMRI LOG+TH (if available) ===
fmri_log_vals = get_upper_values(log_thresholded_connectomes_adrc_fmri_healthy)
print("fMRI (Log+TH)")
print(f"Min: {fmri_log_vals.min():.2f} | Max: {fmri_log_vals.max():.2f} | Mean: {fmri_log_vals.mean():.2f} | Std: {fmri_log_vals.std():.2f}")



# GRAPH METRICS

import networkx as nx
import numpy as np

def compute_clustering_coefficient(matrix):
    G = nx.from_numpy_array(matrix.to_numpy()) #graph
    for u, v, d in G.edges(data=True):
        d["weight"] = matrix.iloc[u, v]
    return nx.average_clustering(G, weight="weight")

def compute_path_length(matrix):
    G = nx.from_numpy_array(matrix.to_numpy())
    for u, v, d in G.edges(data=True):
        weight = matrix.iloc[u, v]
        d["distance"] = 1.0 / weight if weight > 0 else float("inf")
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    try:
        return nx.average_shortest_path_length(G, weight="distance")
    except:
        return float("nan")

def compute_global_efficiency(matrix):
    G = nx.from_numpy_array(matrix.to_numpy())
    for u, v, d in G.edges(data=True):
        d["weight"] = matrix.iloc[u, v]
    return nx.global_efficiency(G)

def compute_local_efficiency(matrix):
    G = nx.from_numpy_array(matrix.to_numpy())
    for u, v, d in G.edges(data=True):
        d["weight"] = matrix.iloc[u, v]
    return nx.local_efficiency(G)

#DTI
# === Ensure the metadata DataFrame has columns for the graph metrics ===
# These metrics will be filled per subject using their log-transformed connectome
df_healthy_adrc["dti_Clustering_Coeff"] = np.nan
df_healthy_adrc["dti_Path_Length"] = np.nan
df_healthy_adrc["dti_Global_Efficiency"] = np.nan
df_healthy_adrc["dti_Local_Efficiency"] = np.nan

# === Loop through each subject and compute graph metrics ===
for subject, matrix_log in log_thresholded_connectomes_adrc_dti_healthy.items():
    try:
        # Compute weighted clustering coefficient (averaged across nodes)
        dti_clustering = compute_clustering_coefficient(matrix_log)

        # Compute average shortest path length (weighted, with inverse distance)
        dti_path = compute_path_length(matrix_log)

        # Compute global efficiency (inverse of average shortest path over all node pairs)
        dti_global_eff = compute_global_efficiency(matrix_log)

        # Compute local efficiency (efficiency of each node's neighborhood)
        dti_local_eff = compute_local_efficiency(matrix_log)

        # Fill the computed values into the corresponding row in the metadata DataFrame
        df_healthy_adrc.loc[df_healthy_adrc["PTID"] == subject, [
            "dti_Clustering_Coeff", "dti_Path_Length", "dti_Global_Efficiency", "dti_Local_Efficiency"
        ]] = [dti_clustering, dti_path, dti_global_eff, dti_local_eff]

    except Exception as e:
        # Catch and report errors (e.g., disconnected graphs or NaNs)
        print(f" Failed to compute metrics for subject {subject}: {e}")

print()



#Functions to not use negative values in path, global and local efficiency

def clean_matrix(matrix):
    arr = matrix.to_numpy().copy()
    arr[arr <= 0] = 0  # Set negative and zero weights to 0
    return pd.DataFrame(arr, index=matrix.index, columns=matrix.columns)


def compute_path_length_noneg(matrix):
    matrix = clean_matrix(matrix)  # clean before graph creation
    G = nx.from_numpy_array(matrix.to_numpy())
    for u, v, d in G.edges(data=True):
        weight = matrix.iloc[u, v]
        d["distance"] = 1.0 / weight if weight > 0 else float("inf")
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    try:
        return nx.average_shortest_path_length(G, weight="distance")
    except:
        return float("nan")


def compute_global_efficiency_noneg(matrix):
    matrix = clean_matrix(matrix)
    G = nx.from_numpy_array(matrix.to_numpy())
    for u, v, d in G.edges(data=True):
        d["weight"] = matrix.iloc[u, v]
    return nx.global_efficiency(G)


def compute_local_efficiency_noneg(matrix):
    matrix = clean_matrix(matrix)
    G = nx.from_numpy_array(matrix.to_numpy())
    for u, v, d in G.edges(data=True):
        d["weight"] = matrix.iloc[u, v]
    return nx.local_efficiency(G)


#FMRI (using no log no th)

# === Add empty columns for fMRI graph metrics in the metadata DataFrame ===
df_healthy_adrc["fmri_Clustering_Coeff"] = np.nan
df_healthy_adrc["fmri_Path_Length"] = np.nan
df_healthy_adrc["fmri_Global_Efficiency"] = np.nan
df_healthy_adrc["fmri_Local_Efficiency"] = np.nan

# === Loop through each subject and compute graph metrics from fMRI connectomes ===
for subject, matrix_fmri in adrc_fmri_connectomes_healthy.items():
    try:
        # Compute weighted clustering coefficient (averaged across nodes)
        fmri_clustering = compute_clustering_coefficient(matrix_fmri)

        # Compute average shortest path length (using inverse of weights)
        fmri_path = compute_path_length_noneg(matrix_fmri)

        # Compute global efficiency
        fmri_global_eff = compute_global_efficiency_noneg(matrix_fmri)

        # Compute local efficiency
        fmri_local_eff = compute_local_efficiency_noneg(matrix_fmri)

        # Fill computed metrics into the DataFrame
        df_healthy_adrc.loc[df_healthy_adrc["PTID"] == subject, [
            "fmri_Clustering_Coeff", "fmri_Path_Length", "fmri_Global_Efficiency", "fmri_Local_Efficiency"
        ]] = [fmri_clustering, fmri_path, fmri_global_eff, fmri_local_eff]

    except Exception as e:
        print(f"Failed to compute fMRI metrics for subject {subject}: {e}")







#Metadata


#Encode sex and apoe
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore

# === Encode SUBJECT_SEX directly (1 and 2) ===
# Optional: map 1 → 0 and 2 → 1 to start from 0
df_healthy_adrc["sex_encoded"] = df_healthy_adrc["SUBJECT_SEX"].map({1: 0, 2: 1})

# === Encode APOE (e.g., "3/4", "4/4", "2/3") ===
# Convert to string in case there are numbers
df_healthy_adrc["genotype"] = LabelEncoder().fit_transform(df_healthy_adrc["APOE"].astype(str))





# ADD BIOMARKERS
# Select biomarkers
biomarker_cols = ["AB40", "AB42", "TTAU", "PTAU181", "NFL", "GFAP"]


# Step 1: Convert AB40 and AB42 to numeric (safe conversion)
df_healthy_adrc["AB40"] = pd.to_numeric(df_healthy_adrc["AB40"], errors='coerce')
df_healthy_adrc["AB42"] = pd.to_numeric(df_healthy_adrc["AB42"], errors='coerce')

# Step 2: Compute the ratio safely
df_healthy_adrc["AB_ratio"] = df_healthy_adrc["AB42"] / df_healthy_adrc["AB40"]
df_healthy_adrc["AB_ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)

# Include the new ratio along with existing biomarkers
biomarker_cols = ["AB40", "AB42", "AB_ratio", "TTAU", "PTAU181", "NFL", "GFAP"]


# Z-score valid values
df_healthy_adrc[biomarker_cols] = df_healthy_adrc[biomarker_cols].apply(
    lambda col: (col - col.mean()) / (col.std() + 1e-8)
)

# Fill NaNs with -10 (clear out-of-distribution marker)
df_healthy_adrc[biomarker_cols] = df_healthy_adrc[biomarker_cols].fillna(-10)











#Nomalize graph metrics with zscore
# === Define which columns are dti graph-level metrics ===
dti_metrics = ["dti_Clustering_Coeff", "dti_Path_Length", "dti_Global_Efficiency", "dti_Local_Efficiency"]

# === Apply z-score normalization across subjects ===
df_healthy_adrc[dti_metrics] = df_healthy_adrc[dti_metrics].apply(zscore)



# === Define graph metric columns for fMRI ===

fmri_metrics = ["fmri_Clustering_Coeff", "fmri_Path_Length", "fmri_Global_Efficiency", "fmri_Local_Efficiency"]

# === Apply z-score normalization across subjects ===
df_healthy_adrc[fmri_metrics] = df_healthy_adrc[fmri_metrics].apply(zscore)





#Build global feature tensors

import torch

# === Demographic tensor per subject: [sex_encoded, genotype] ===
subject_to_demographic_tensor = {
    row["PTID"]: torch.tensor([
        row["sex_encoded"],
        row["genotype"]
    ], dtype=torch.float)
    for _, row in df_healthy_adrc.iterrows()
}

# === DTI graph metrics tensor: [Clustering, Path Length, Global Eff., Local Eff.] ===
subject_to_dti_graphmetrics_tensor = {
    row["PTID"]: torch.tensor([
        row["dti_Clustering_Coeff"],
        row["dti_Path_Length"],
        row["dti_Global_Efficiency"],
        row["dti_Local_Efficiency"]
    ], dtype=torch.float)
    for _, row in df_healthy_adrc.iterrows()
}

# === fMRI graph metrics tensor: [Clustering, Path Length, Global Eff., Local Eff.] ===
subject_to_fmri_graphmetrics_tensor = {
    row["PTID"]: torch.tensor([
        row["fmri_Clustering_Coeff"],
        row["fmri_Path_Length"],
        row["fmri_Global_Efficiency"],
        row["fmri_Local_Efficiency"]
    ], dtype=torch.float)
    for _, row in df_healthy_adrc.iterrows()
}




# Step 5: Rebuild tensor dictionary
subject_to_biomarker_tensor = {
    row["PTID"]: torch.tensor(row[biomarker_cols].values.astype(np.float32))
    for _, row in df_healthy_adrc.iterrows()
}



#Convert ADRC DTI matrices to PyTorch Geometric graph objects

import torch
from torch_geometric.data import Data

# === Device setup ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")




# 1 -> DTI

# === Create list to store graph data objects
graph_data_list_adrc_dti = []


# === Create mapping: subject ID → age
subject_to_age = df_healthy_adrc.set_index("PTID")["SUBJECT_AGE_SCREEN"].to_dict()



# === Iterate over each healthy subject's processed matrix ===
for subject, matrix_log in log_thresholded_connectomes_adrc_dti_healthy.items():
    try:
        # Skip if required components are missing
        if subject not in subject_to_demographic_tensor:
            continue
        if subject not in subject_to_dti_graphmetrics_tensor:
            continue
        if subject not in subject_to_biomarker_tensor:
            continue
        if subject not in normalized_node_features_dict:
            continue

        # === Convert connectome matrix to edge_index, edge_attr, node_features
        edge_index, edge_attr, node_features = matrix_to_graph(
            matrix_log, device, subject, normalized_node_features_dict
        )

        if subject not in subject_to_age:
            continue
        age = torch.tensor([subject_to_age[subject]], dtype=torch.float)


        
        # === Concatenate demographic + graph metrics to form global features
        demo_tensor = subject_to_demographic_tensor[subject]   # [2]
        biomarker_tensor = subject_to_biomarker_tensor[subject]    # [7]
        dti_tensor = subject_to_dti_graphmetrics_tensor[subject]     # [4]
        
        global_feat_dti = torch.cat([demo_tensor,   biomarker_tensor, dti_tensor], dim=0)  



        # === Create graph object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=age,
            global_features=global_feat_dti.unsqueeze(0)
        )
        data.subject_id = subject
        graph_data_list_adrc_dti.append(data)
        
        # DEBUG: print to verify
        print(f"ADDED → Subject: {subject} | Assigned Age: {age.item()}")


    except Exception as e:
        print(f" Failed to process subject {subject}: {e}")

# === Preview one example graph ===
sample_graph = graph_data_list_adrc_dti[0]
print("=== ADRC Sample Graph ===")
print(f"Node feature shape (x): {sample_graph.x.shape}")         
print(f"Edge index shape: {sample_graph.edge_index.shape}")     
print(f"Edge attr shape: {sample_graph.edge_attr.shape}")       
print(f"Global features shape: {sample_graph.global_features.shape}")  
print(f"Target age (y): {sample_graph.y.item()}")                

print("\nFirst 5 edge weights:")
print(sample_graph.edge_attr[:5])

print("\nGlobal features vector:")
print(sample_graph.global_features)
print()



import matplotlib.pyplot as plt

ages = [data.y.item() for data in graph_data_list_adrc_dti]
plt.hist(ages, bins=20)
plt.title("Distribution of Real Ages")
plt.xlabel("Age")
plt.ylabel("Count")
plt.grid(True)

plt.savefig(os.path.join(output_dir, "Distribution_real_ages.png"), dpi=300)
plt.close()




# 2 -> fMRI

# === Create list to store graph data objects
graph_data_list_adrc_fmri = []


# === Create mapping: subject ID → age
subject_to_age = df_healthy_adrc.set_index("PTID")["SUBJECT_AGE_SCREEN"].to_dict()



# === Iterate over each healthy subject's processed matrix ===
for subject, matrix_log in adrc_fmri_connectomes_healthy.items():
    try:
        # Skip if required components are missing
        if subject not in subject_to_demographic_tensor:
            continue
        
        if subject not in subject_to_fmri_graphmetrics_tensor:
            continue
        
        if subject not in subject_to_biomarker_tensor:
            continue
        
        if subject not in normalized_node_features_dict:
            continue

        # === Convert connectome matrix to edge_index, edge_attr, node_features
        edge_index, edge_attr, node_features = matrix_to_graph(
            matrix_log, device, subject, normalized_node_features_dict
        )

        if subject not in subject_to_age:
            continue
        age = torch.tensor([subject_to_age[subject]], dtype=torch.float)


       
        # === Concatenate demographic + graph metrics to form global features
        #demo_tensor = subject_to_demographic_tensor[subject]   # [2]
        fmri_tensor = subject_to_fmri_graphmetrics_tensor[subject]     # [4]
        #biomarker_tensor = subject_to_biomarker_tensor[subject]    # [7]
        
        
        global_feat_fmri = fmri_tensor  # [4]



        # === Create graph object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=age,
            global_features=global_feat_fmri.unsqueeze(0)
        )
        data.subject_id = subject
        graph_data_list_adrc_fmri.append(data)
        
        # DEBUG: print to verify
        print(f"ADDED → Subject: {subject} | Assigned Age: {age.item()}")


    except Exception as e:
        print(f" Failed to process subject {subject}: {e}")








######################  DEFINE MODEL  #########################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm

class DualGATv2_EarlyFusion(nn.Module):
    def __init__(self):
        super(DualGATv2_EarlyFusion, self).__init__()

        # === Node Embedding shared across modalities (assumes same node features for DTI/fMRI) ===
        self.node_embed = nn.Sequential(
            nn.Linear(2, 64),  # Assumes 2 node features (e.g., FA, Volume)
            nn.ReLU(),
            nn.Dropout(0.15)
        )

        # === GATv2 backbone for DTI ===
        self.gnn_dti_1 = GATv2Conv(64, 16, heads=8, concat=True, edge_dim=1)
        self.bn_dti_1 = BatchNorm(128)

        self.gnn_dti_2 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn_dti_2 = BatchNorm(128)

        self.gnn_dti_3 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn_dti_3 = BatchNorm(128)

        self.gnn_dti_4 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn_dti_4 = BatchNorm(128)

        # === GATv2 backbone for fMRI ===
        self.gnn_fmri_1 = GATv2Conv(64, 16, heads=8, concat=True, edge_dim=1)
        self.bn_fmri_1 = BatchNorm(128)

        self.gnn_fmri_2 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn_fmri_2 = BatchNorm(128)

        self.gnn_fmri_3 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn_fmri_3 = BatchNorm(128)

        self.gnn_fmri_4 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn_fmri_4 = BatchNorm(128)

        self.dropout = nn.Dropout(0.3)

        # === GLOBAL FEATURE BRANCHES (shared) ===
        self.meta_head = nn.Sequential(
            nn.Linear(2, 16),  # sex, genotype
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 16),
            nn.ReLU()
        )

        self.graph_dti_head = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        self.graph_fmri_head = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        
        # Biomarkers: 6 features
        self.bio_head = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 32),
            nn.ReLU()
          )




        # Final MLP after concatenating DTI + fMRI + global
        self.fc = nn.Sequential(
            nn.Linear(128 + 128 + 16+32+32+32 , 128),  # 128 DTI + 128 fMRI + metadata + graph metrics dti + graph metrics fmri+  biomarkers
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data_dti, data_fmri):
        # === Node features shared ===
        x_dti = self.node_embed(data_dti.x)
        x_fmri = self.node_embed(data_fmri.x)

        # === DTI Stream ===
        x_dti = self.gnn_dti_1(x_dti, data_dti.edge_index, data_dti.edge_attr)
        x_dti = self.bn_dti_1(x_dti)
        x_dti = F.relu(x_dti)

        x_dti = F.relu(self.bn_dti_2(self.gnn_dti_2(x_dti, data_dti.edge_index, data_dti.edge_attr)) + x_dti)
        x_dti = F.relu(self.bn_dti_3(self.gnn_dti_3(x_dti, data_dti.edge_index, data_dti.edge_attr)) + x_dti)
        x_dti = F.relu(self.bn_dti_4(self.gnn_dti_4(x_dti, data_dti.edge_index, data_dti.edge_attr)) + x_dti)
        x_dti = self.dropout(x_dti)
        x_dti = global_mean_pool(x_dti, data_dti.batch)

        # === fMRI Stream ===
        x_fmri = self.gnn_fmri_1(x_fmri, data_fmri.edge_index, data_fmri.edge_attr)
        x_fmri = self.bn_fmri_1(x_fmri)
        x_fmri = F.relu(x_fmri)

        x_fmri = F.relu(self.bn_fmri_2(self.gnn_fmri_2(x_fmri, data_fmri.edge_index, data_fmri.edge_attr)) + x_fmri)
        x_fmri = F.relu(self.bn_fmri_3(self.gnn_fmri_3(x_fmri, data_fmri.edge_index, data_fmri.edge_attr)) + x_fmri)
        x_fmri = F.relu(self.bn_fmri_4(self.gnn_fmri_4(x_fmri, data_fmri.edge_index, data_fmri.edge_attr)) + x_fmri)
        x_fmri = self.dropout(x_fmri)
        x_fmri = global_mean_pool(x_fmri, data_fmri.batch)





        # === Global features (same for both) ===
        global_feat = torch.cat([data_dti.global_features, data_fmri.global_features], dim=1).to(data_dti.x.device).squeeze(1)  #.to(data_dti.x.device) makes sure global_feat is moved to the same device as input tensors

        # [B, 13]  (meta +  bio +DTI  )
        # [B, 4]   (only fMRI metrics)
         # → [B, 17]



        meta_embed = self.meta_head(global_feat[:, 0:2]) # all rows, first two columns
        bio_embed = self.bio_head(global_feat[:, 2:9])     # 7 biomarkers
        graph_dti_embed = self.graph_dti_head(global_feat[:, 9:13]) # all rows, columns 4 (dti_Clustering, dti_PathLength, dti_GlobalEff, dti_LocalEff )
       
        graph_fmri_embed = self.graph_fmri_head(global_feat[:, 13:17])  # all rows, columns   (fmri_Clustering, fmri_PathLength, fmri_GlobalEff, fmri_LocalEff)
        

        
        
        
        global_embed = torch.cat([meta_embed, bio_embed, graph_dti_embed, graph_fmri_embed   ], dim=1)




        # === Fusion and prediction ===
        x = torch.cat([x_dti, x_fmri, global_embed], dim=1)
        out = self.fc(x)

        return out







# Create lookup dictionaries from subject_id to graph
dti_dict = {g.subject_id: g for g in graph_data_list_adrc_dti}
fmri_dict = {g.subject_id: g for g in graph_data_list_adrc_fmri}

# Keep only common subjects
common_subjects = sorted(set(dti_dict.keys()) & set(fmri_dict.keys()))

# Build aligned list of (DTI, fMRI) graph pairs
graph_data_list_adrc_bimodal = [(dti_dict[pid], fmri_dict[pid]) for pid in common_subjects]

print(f"Total bimodal subjects: {len(graph_data_list_adrc_bimodal)}")



#collate_fn=collate_bimodal to properly batch pairs of DTI and fMRI graphs during training and evaluation. 
#It ensures each modality is grouped separately into Batch objects for input to the model.

def collate_bimodal(batch):
    data_dti_list, data_fmri_list = zip(*batch)  # separa los pares
    return Batch.from_data_list(data_dti_list), Batch.from_data_list(data_fmri_list)




    
from torch.optim import Adam
from torch_geometric.loader import DataLoader  # Usamos el DataLoader de torch_geometric

def train(model, train_loader, optimizer, criterion):
    model.train()  # Set the model to training mode
    total_loss = 0  # Initialize the total loss for the epoch

    # Iterate through the training data loader
    for data_dti, data_fmri in train_loader:
        data_dti = data_dti.to(device)  # Move DTI graph to GPU
        data_fmri = data_fmri.to(device)  # Move fMRI graph to GPU

        optimizer.zero_grad()  # Clear previous gradients

        # Forward pass through the model with both DTI and fMRI inputs
        output = model(data_dti, data_fmri).view(-1)

        # Compute loss using the target age (assumed same in both DTI/fMRI)-> the target, the age is the same in both
        loss = criterion(output, data_dti.y)

        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update model weights

        total_loss += loss.item()  # Accumulate batch loss

    return total_loss / len(train_loader)  # Return average loss for the epoch




def evaluate(model, test_loader, criterion):
    model.eval()  # Set model to evaluation mode
    total_loss = 0  # Initialize total loss

    with torch.no_grad():  # Disable gradient computation
        for data_dti, data_fmri in test_loader:
            # Move each modality batch to the device
            data_dti = data_dti.to(device)
            data_fmri = data_fmri.to(device)

            # Forward pass through the model
            output = model(data_dti, data_fmri).view(-1)

            # Compute loss using DTI target (same age for both modalities)
            loss = criterion(output, data_dti.y)

            total_loss += loss.item()

    return total_loss / len(test_loader)





import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Training parameters
epochs = 300
patience = 40
k = 7  # Folds
batch_size = 6
repeats_per_fold = 10




# Initialize loss tracking
all_train_losses_bimodal = []
all_test_losses_bimodal = []
all_early_stopping_epochs_bimodal = []

# Get subject IDs
graph_subject_ids_bimodal = [pair[0].subject_id for pair in graph_data_list_adrc_bimodal]
df_filtered = df_healthy_adrc[df_healthy_adrc["PTID"].isin(graph_subject_ids_bimodal)].copy()
df_filtered = df_filtered.set_index("PTID").loc[graph_subject_ids_bimodal].reset_index()

# Create stratification bins for age
ages = df_filtered["SUBJECT_AGE_SCREEN"].to_numpy()
age_bins = pd.qcut(ages, q=5, labels=False)
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Main cross-validation loop
for fold, (train_idx, test_idx) in enumerate(skf.split(graph_data_list_adrc_bimodal, age_bins)):
    print(f"\n--- Bimodal Fold {fold+1}/{k} ---")

    train_data = [graph_data_list_adrc_bimodal[i] for i in train_idx]
    test_data = [graph_data_list_adrc_bimodal[i] for i in test_idx]

    fold_train_losses = []
    fold_test_losses = []

    for repeat in range(repeats_per_fold):
        print(f'  > Repeat {repeat+1}/{repeats_per_fold}')
        early_stop_epoch = None

        seed_everything(42 + repeat)

        #Dataloaders with collate_fn
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_bimodal)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_bimodal)


        model = DualGATv2_EarlyFusion().to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        criterion = torch.nn.SmoothL1Loss(beta=1)

        train_losses = []
        test_losses = []
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = train(model, train_loader, optimizer, criterion)
            test_loss = evaluate(model, test_loader, criterion)

            train_losses.append(train_loss)
            test_losses.append(test_loss)

            if test_loss < best_loss:
                best_loss = test_loss
                patience_counter = 0
                torch.save(model.state_dict(), f"bimodal_model_fold_{fold+1}_rep_{repeat+1}.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    early_stop_epoch = epoch + 1
                    print(f"    Early stopping triggered at epoch {early_stop_epoch}.")
                    break

            scheduler.step()

        # Save early stop epoch
        if early_stop_epoch is None:
            early_stop_epoch = epochs
        all_early_stopping_epochs_bimodal.append((fold + 1, repeat + 1, early_stop_epoch))

        fold_train_losses.append(train_losses)
        fold_test_losses.append(test_losses)

    all_train_losses_bimodal.append(fold_train_losses)
    all_test_losses_bimodal.append(fold_test_losses)








#################  LEARNING CURVE GRAPH (MULTIPLE REPEATS)  ################

plt.figure(figsize=(10, 6))

# Plot average learning curves across all repeats for each fold
for fold in range(k):
    for rep in range(repeats_per_fold):
        plt.plot(all_train_losses_bimodal[fold][rep],
                 label=f'Train Loss - Fold {fold+1} Rep {rep+1}', linestyle='dashed', alpha=0.5)
        plt.plot(all_test_losses_bimodal[fold][rep],
                 label=f'Test Loss - Fold {fold+1} Rep {rep+1}', alpha=0.5)

plt.xlabel("Epochs")
plt.ylabel("Smooth L1 Loss")
plt.title("Learning Curves - Bimodal Model (All Repeats)")
plt.legend(loc="upper right", fontsize=7)
plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "All_folds_reps_learningCurva_ADRC_bimodal.png"), dpi=300)
plt.close()



# ==== LEARNING CURVE PLOT (MEAN ± STD) FOR BIMODAL MODEL ====

import numpy as np
import matplotlib.pyplot as plt

# Compute mean and std for each epoch across all folds and repeats
avg_train_bimodal = []
avg_test_bimodal = []

for epoch in range(epochs):
    epoch_train = []
    epoch_test = []
    for fold in range(k):
        for rep in range(repeats_per_fold):
            if epoch < len(all_train_losses_bimodal[fold][rep]):
                epoch_train.append(all_train_losses_bimodal[fold][rep][epoch])
                epoch_test.append(all_test_losses_bimodal[fold][rep][epoch])
    avg_train_bimodal.append((np.mean(epoch_train), np.std(epoch_train)))
    avg_test_bimodal.append((np.mean(epoch_test), np.std(epoch_test)))

# Unpack into arrays
train_mean, train_std = zip(*avg_train_bimodal)
test_mean, test_std = zip(*avg_test_bimodal)

# Plot
plt.figure(figsize=(10, 6))

plt.plot(train_mean, label="Train Mean", color="blue")
plt.fill_between(range(epochs), np.array(train_mean) - np.array(train_std),
                 np.array(train_mean) + np.array(train_std), color="blue", alpha=0.3)

plt.plot(test_mean, label="Test Mean", color="orange")
plt.fill_between(range(epochs), np.array(test_mean) - np.array(test_std),
                 np.array(test_mean) + np.array(test_std), color="orange", alpha=0.3)

plt.xlabel("Epoch")
plt.ylabel("Smooth L1 Loss")
plt.title("Learning Curve Bimodal (Mean ± Std Across All Folds/Repeats)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "Mean_LearningCurve_ADRC_bimodal.png"), dpi=300)
plt.close()







#####################  PREDICTION & METRIC ANALYSIS ACROSS FOLDS/REPEATS — ADRC BIMODAL  #####################
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
import os

# === Storage containers (per-fold lists + master lists) ===
fold_mae_list_bimodal, fold_rmse_list_bimodal, fold_r2_list_bimodal = [], [], []
all_y_true_bimodal, all_y_pred_bimodal, all_subject_ids_bimodal     = [], [], []
fold_tag_bimodal,   repeat_tag_bimodal                               = [], []

# (Re)construct StratifiedKFold on the same age bins used for training
skf_bimodal = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
ages_bimodal     = df_filtered["SUBJECT_AGE_SCREEN"].to_numpy()
age_bins_bimodal = pd.qcut(ages_bimodal, q=5, labels=False)

for fold, (train_idx, test_idx) in enumerate(skf_bimodal.split(graph_data_list_adrc_bimodal, age_bins_bimodal)):
    print(f"\n--- Evaluating BIMODAL Fold {fold+1}/{k} ---")

    test_data   = [graph_data_list_adrc_bimodal[i] for i in test_idx]
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_bimodal)

    # Per-fold temporary containers
    repeat_mae, repeat_rmse, repeat_r2 = [], [], []

    for rep in range(repeats_per_fold):
        print(f"  > Repeat {rep+1}/{repeats_per_fold}")

        model = DualGATv2_EarlyFusion().to(device)
        model.load_state_dict(torch.load(f"bimodal_model_fold_{fold+1}_rep_{rep+1}.pt"))
        model.eval()

        y_true_rep, y_pred_rep, subj_ids_rep = [], [], []

        with torch.no_grad():
            for data_dti, data_fmri in test_loader:
                data_dti  = data_dti.to(device)
                data_fmri = data_fmri.to(device)

                preds = model(data_dti, data_fmri).view(-1)
                trues = data_dti.y.view(-1)

                y_pred_rep.extend(preds.cpu().tolist())
                y_true_rep.extend(trues.cpu().tolist())
                subj_ids_rep.extend([str(s) for s in data_dti.subject_id])

        # === Metrics for this repeat ===
        mae  = mean_absolute_error(y_true_rep, y_pred_rep)
        rmse = mean_squared_error(y_true_rep, y_pred_rep, squared=False)
        r2   = r2_score(y_true_rep, y_pred_rep)

        repeat_mae.append(mae)
        repeat_rmse.append(rmse)
        repeat_r2.append(r2)

        # === Append to master lists ===
        all_y_true_bimodal.extend(y_true_rep)
        all_y_pred_bimodal.extend(y_pred_rep)
        all_subject_ids_bimodal.extend(subj_ids_rep)
        fold_tag_bimodal.extend([fold + 1] * len(y_true_rep))
        repeat_tag_bimodal.extend([rep  + 1] * len(y_true_rep))

    # === Store per-fold results ===
    fold_mae_list_bimodal.append(repeat_mae)
    fold_rmse_list_bimodal.append(repeat_rmse)
    fold_r2_list_bimodal.append(repeat_r2)

    print(f">> Fold {fold+1} | "
          f"MAE:  {np.mean(repeat_mae):.2f} ± {np.std(repeat_mae):.2f} | "
          f"RMSE: {np.mean(repeat_rmse):.2f} ± {np.std(repeat_rmse):.2f} | "
          f"R²:   {np.mean(repeat_r2):.2f} ± {np.std(repeat_r2):.2f}")

# === Global aggregates ===
all_maes  = np.concatenate(fold_mae_list_bimodal)
all_rmses = np.concatenate(fold_rmse_list_bimodal)
all_r2s   = np.concatenate(fold_r2_list_bimodal)

print("\n================== FINAL METRICS BIMODAL ==================")
print(f"Global MAE:  {all_maes.mean():.2f} ± {all_maes.std():.2f}")
print(f"Global RMSE: {all_rmses.mean():.2f} ± {all_rmses.std():.2f}")
print(f"Global R²:   {all_r2s.mean():.2f} ± {all_r2s.std():.2f}")
print("===========================================================\n")

# === Save full prediction CSV ===
df_preds_bimodal = pd.DataFrame({
    "Subject_ID":   all_subject_ids_bimodal,
    "Real_Age":     all_y_true_bimodal,
    "Predicted_Age":all_y_pred_bimodal,
    "Fold":         fold_tag_bimodal,
    "Repeat":       repeat_tag_bimodal
})

csv_path = os.path.join(output_dir, "cv_predictions_adrc_bimodal.csv")
df_preds_bimodal.to_csv(csv_path, index=False)
print(f" CSV saved to: {csv_path}\n")

