
### 0_PearsonCorrelation.py

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

