# Classification Results & Validation Methodology

## Overview
This directory contains the final validation metrics and performance reports for the Rule-Based Image Classification pipeline. The results demonstrate the effectiveness of the "Human-in-the-Loop" reference refinement strategy.

## Validation Toolchain
The following utilities from the `Codes/` directory were employed to generate these metrics:

### 1. Ground Truth Generation (`Image_Labelling.py`)
To mathematically calculate accuracy, the system requires a "Golden Set" where the true category of every image is known. 
*   **Function**: This script standardizes filenames into a `Label_Index` format (e.g., `Building_001.jpg`).
*   **Usage**: It allows `Reports.py` to extract the "True Label" directly from the filename string.

### 2. Metric Calculation (`Reports.py`)
This script compares the physical location of an image (Predicted Class) against its filename (True Class).
*   **Output**: It generates Confusion Matrices, Precision/Recall scores, and Misclassification logs found in this directory.

---

## Iterative Improvement Process

### Phase 1: Initial Baseline
*   **Location**: `../Data/classification_reports`
*   **Documentation**: [Data Staging & Baseline Metrics](../Data/Data.md)
*   **Method**: Initial classification was performed using a randomly selected set of 5-10 reference images per category.
*   **Outcome**: These preliminary results highlighted specific confusion points (e.g., distinguishing between 'Logos' and 'Icons').

### Phase 2: Refinement (Current Results)
*   **Location**: `../Results`
*   **Optimization**: Based on the misclassification logs from Phase 1, the `Reference_Images_final` dataset was curated. This involved:
    1.  Removing outlier reference images that caused false positives.
    2.  Adding "Hard Negative" examples to the reference set to improve boundary decision making.
*   **Outcome**: The rule-based classifier's discrimination ability was significantly enhanced without changing the underlying code logic.

---

## Final Performance Summary

The following metrics represent the performance of the classifier on the finalized test set (N=460 images) using the optimized reference images.

| Metric | Value |
| :--- | :--- |
| **Overall Accuracy** | **94.13%** |
| Total Images | 460 |
| Correctly Classified | 433 |
| Misclassified | 27 |
| Weighted F1 Score | 0.9428 |

*For detailed breakdown by category, refer to `classification_metrics_*.csv` and `confusion_matrix_*.csv` in the `classification_reports_final` directory.*
