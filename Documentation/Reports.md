# Reports.py - Classification Accuracy & Validation

## 1. Overview
`Reports.py` provides the analytical framework for the project. While `Clean_Image.py` performs the classification, `Reports.py` measures how *well* it performed. This is essential for tuning the threshold parameters in the rule-based system.

## 2. Validation Methodology
The script generates specific metrics by comparing the **Predicted Class** (the folder where the script placed an image) against the **True Class** (derived from the strictly formatted filename, e.g., `Logo_12.jpg` implies a true label of 'Logo').

### Generated Metrics:
1.  **Confusion Matrix**: A tabular visualization showing the intersection of predictions. It highlights specific confusion pairs (e.g., "How many *Diagrams* were mistaken for *Icons*?").
2.  **Class-wise Accuracy**: Precision, Recall, and F1-Scores for each category.
3.  **Misclassification Log**: A detailed list of every specific file that was sorted incorrectly.

## 3. Interpreting Results
Users should refer to this module's output to make decisions about the **Reference Images**:
*   **Low Precision**: Indicates the reference images for that class are too broad (capturing too many false positives).
*   **Low Recall**: Indicates the reference images are too specific (missing valid instances).

## 4. Accessing Reports
The output logs and CSV reports are generated in the `Results/` directory.
*   **Files**: Look for `classification_metrics.csv` and `confusion_matrix.csv`.
*   **Note**: The accuracy figures cited in the final documentation are achieved by iteratively running this report and adjusting the `Reference_Images` folders until the desired F1-score is reached.
