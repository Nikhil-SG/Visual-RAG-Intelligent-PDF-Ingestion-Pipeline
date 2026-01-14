# Analysis of Rule-Based Classification on Diverse Datasets

## 1. Objective
The primary objective of this analysis module was to stress-test the **Rule-Based Image Classifier** (`Codes/Clean_Image.py`) against a highly diverse, semantic dataset. While the classifier performs exceptionally well on distinct document elements (icons vs. charts), this experiment aimed to evaluate its limitations when applied to general object recognition tasks, specifically animal classification.

## 2. Dataset & Methodology
**Dataset Source**: [Animal Image Dataset (90 Different Animals)](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals/data)

### Workflow:
1.  **Exploratory Data Analysis (EDA)**: The notebook [`Animals_EDA.ipynb`](Animals_EDA.ipynb) was used to inspect the dataset structure, distribution, and prepare the file naming conventions for processing.
2.  **Reference Creation**: A semi-automated selection process randomly chose 6 images per category to serve as the "Reference Set" for the rule-based learner.
3.  **Classification**: The `Clean_Image.py` script attempted to classify 5,400 images into 90 categories using strictly computer vision feature vectors (Color, Texture, Edges).

## 3. Performance Results
The rule-based approach demonstrated **poor performance** on this specific task, highlighting the distinct boundary between "Document Analysis" and "Semantic Object Recognition."

| Metric | Value |
| :--- | :--- |
| **Overall Accuracy** | **14.09%** |
| Total Categories | 90 |
| Total Images | 5,400 |
| Correctly Classified | 761 |
| Misclassified | 4,639 |
| Macro F1 Score | 0.1452 |

*Full metrics available in: `classification_reports/summary_statistics_*.csv`*

## 4. Root Cause Analysis: Why it Failed
The failure of the rule-based approach on animal datasets is expected and pedagogically significant. It stems from the nature of the features being extracted.

### A. Feature Invariance vs. Variance
The `Clean_Image.py` script relies on **Classical Computer Vision** features:
*   **Color Histograms**: Effective for distinguishing a blue "Facebook Logo" from a black and white "Text Block". However, a "Black Bear" and a "Black Cat" have nearly identical color histograms, leading to confusion.
*   **Edge Density**: Effective for distinguishing a simple "Icon" from a complex "Photograph". However, the fur of a wolf and the fur of a dog generate similar edge density profiles.
*   **Aspect Ratio**: Crucial for distinguishing "Banners" (wide) from "Headshots" (square). Completely irrelevant for animals, as their aspect ratio changes based on pose and photo cropping.

### B. Semantic Gap
Rule-based systems lack **Semantic Understanding**.
*   They see "Orange patch with stripes" (Texture/Color).
*   They do not see "Ears," "Snouts," or "Tails."
*   **Intra-class Variance**: A generic rule cannot account for a dog sitting, running, or swimming. To a rule-based system, these look like three different objects.
*   **Inter-class Similarity**: A wolf and a husky look geometrically identical to a feature extractor, distinguishing them requires learning subtle, non-linear patterns.

## 5. Recommendations for General Object Recognition
For datasets of this nature (high semantic variance), **Deep Learning** is non-negotiable.

*   **Convolutional Neural Networks (CNNs)**: Architectures like **ResNet** or **EfficientNet** learn hierarchical features (Lines -> Shapes -> Eyes -> Faces) that are invariant to pose and lighting.
*   **Vision Transformers (ViTs)**: State-of-the-art models that capture global context better than CNNs.
*   **Transfer Learning**: Using a pre-trained model (like ImageNet) would likely achieve 90%+ accuracy on this animal dataset with minimal fine-tuning.

## 6. Conclusion
The Rule-Based Classifier (`Clean_Image.py`) is a specialized tool optimized for **Document Layout Analysis** (distinguishing structural elements in PDFs). It is **not** a general-purpose Object Detector.

**Why it works for PDFs:**
*   **Repetition**: Corporate documents often reuse the exact same assets (logos, social media icons) across pages. A feature match is often 100% exact.
*   **Distinct Classes**: In a PDF, a "Photograph" looks mathematically very different from a "Vector Icon."

This experiment confirms that while it is highly efficient (fast, private, CPU-only) for cleaning RAG datasets, it cannot replace Neural Networks for semantic classification tasks.

## 7. Technical Documentation
For a detailed breakdown of the internal logic of the classifier used in this experiment:
*   [Technical Specification (Markdown)](README_Clean_Image.md)
