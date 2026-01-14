# Data Staging & Initial Extraction

## Overview
This directory acts as the staging area for the RAG pipeline. It contains the raw outputs from the PDF ingestion process and the initial, experimental setup for image classification. This data represents the "Baseline" phase of the project, where random sampling was used to establish initial performance metrics.

## Directory Structure & Contents

### 1. Raw Extraction Output
*   **`MD/`**: Contains the raw Markdown files extracted from the source PDFs via `PDF2MD.py`. All text and image references are preserved here in their original order.
*   **`Images/`**: Contains the complete set of binary images extracted from the PDF streams. These are the raw assets before any filtering or classification.

### 2. Classification Setup (Baseline)
*   **`Reference_Images/`**: This folder represents the "Human-in-the-Loop" configuration.
    *   **Methodology**: A human annotator reviewed the raw `Images/` and manually created subdirectories (classes) and selected a small number of random samples to serve as references.
    *   **Purpose**: These images define the ground truth for the rule-based classifier.

### 3. Processing Results
*   **`classified_images/`**: The output of the `Clean_Image.py` script. Images from the source folder were sorted into these categories based on their similarity to the `Reference_Images`.
*   **`classification_reports/`**: Performance metrics derived from this specific configuration.

---

## Baseline Performance Metrics

For this initial phase, the reference images were selected randomly without aggressive optimization (e.g., no "hard negatives" or outlier removal). Despite this minimal setup, the rule-based approach achieved respectable baseline accuracy.

| Metric | Value |
| :--- | :--- |
| **Overall Accuracy** | **81.30%** |
| Total Images | 460 |
| Correctly Classified | 374 |
| Misclassified | 86 |
| Weighted F1 Score | 0.8342 |

*See `classification_reports/summary_statistics_*.csv` for full details.*

## Progression
While an accuracy of 81.3% is a strong starting point, it resulted in 86 misclassified images (false positives/negatives). These error cases were analyzed to curate a superior reference set.

For the optimized results and the final high-precision configuration (achieving >94% accuracy), please refer to the **Results Module**:
*   [Results & Final Validation](../Results/Results.md)
