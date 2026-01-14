# Visual-RAG: Intelligent PDF Ingestion Pipeline

> Developed a computer-vision based data ingestion pipeline for RAG systems that reduces token usage by eliminating non-semantic visual noise (icons, decorative elements) using custom signal processing algorithms (Edge Density, Texture Analysis) rather than expensive external APIs.

## Project Overview
This project implements a robust **Retrieval-Augmented Generation (RAG)** pipeline designed specifically for ingesting, processing, and cleaning data from PDF documents (such as [university prospectuses](Content/Content.md)). 

The primary challenge in RAG systems dealing with rich documents is the "Garbage In, Garbage Out" problem. Standard extractors often treat headers, footers, logos, icons, and decorative elements as meaningful content. When fed into an LLM context window—especially in multimodal models—these irrelevant images consume valuable tokens and degrade the quality of the model's retrieval and reasoning.

This solution optimizes the data ingestion process by:
1.  **Extracting** content with high fidelity from PDFs.
2.  **Classifying** visual elements to distinguish between "Content" (diagrams, photos) and "Noise" (icons, logos).
3.  **Filming/Cleaning** the dataset to retain only information-rich assets.

## Key Features
*   **Token Optimization**: Reduces context window usage by removing hundreds of non-informative images.
*   **Structure Preservation**: Maintains the logical flow of text and their associated images via Markdown.
*   **Privacy & Efficiency**: Uses a local, rule-based Computer Vision approach for image classification, avoiding the latency and cost of sending every image to an API for filtering.

---

## Workflow Architecture

The pipeline operates in four distinct stages:

### 1. Ingestion (`PDF2MD.py`)
*   **Documentation**: [See detailed Technical Specification](Documentation/PDF2MD.md)
*   **RAG Solution**: Unlike generic text-only extractors, this module treats PDFs as visual documents. It utilizes `PyMuPDF` to traverse documents page-by-page.
*   **Mechanism**: It extracts text into structured Markdown while simultaneously extracting binary image streams.
*   **Output**: A raw Markdown file with placeholder links and extracted images. [See Data Structure](Data/Data.md).

### 2. Classification (`Clean_Image.py`)
*   **Documentation**: [See detailed Technical Specification](Documentation/Clean_Image.md)
*   **Logic**: A rule-based classifier that uses signal processing techniques (Color Histograms, Edge Density, Texture Variance).
*   **Reference System**: We employ a "Human-in-the-Loop" strategy. A user creates a small set of "Reference Images" (exemplars of Logos, Icons, Content, etc.). The script compares new unknown images against these references using weighted similarity metrics.
*   **Benefit**: This eliminates the need for training black-box neural networks and allows for easy adjustment of categories.

### 3. Validation (`Reports.py` & `Image_Labelling.py`)
*   **Documentation**: [Reports Spec](Documentation/Reports.md) | [Labelling Spec](Documentation/Image_Labelling.md)
*   **Purpose**: These utilities allow for the quantitative assessment of the classification accuracy.
*   **Note**: These are primarily used during the tuning phase. `Image_Labelling.py` normalizes filenames for ground-truth creation, and `Reports.py` generates confusion matrices to ensure high precision in filtering.
*   **Results**: See [Validation Results & Analysis](Results/Results.md) to verify the accuracy achieved by the current reference set.

### 4. Finalization (`cleanup_classified_images.py`)
*   **Documentation**: [See detailed Technical Specification](Documentation/cleanup_classified_images.md)
*   **Action**: This is the commit step. Based on the classification, it performs a physical cleanup.
*   **Synchronization**: It deletes irrelevant image files from the disk and computationally removes their reference tags from the Markdown files.
*   **Result**: Validated, clean Markdown files in `Finalized/MD` and relevant images in `Finalized/Images`. [See Cleanup Protocol](Finalized/Finalized.md).

---

## Project Structure

```text
├── Codes/
│   ├── PDF2MD.py                   # Step 1: Extracts Text & Images
│   ├── Clean_Image.py              # Step 2: Classifies Images
│   ├── cleanup_classified_images.py # Step 4: Deletes Junk & Updates MD
│   ├── Image_Labelling.py          # Util: Renames files for dataset creation
│   └── Reports.py                  # Util: Generates accuracy metrics
├── Content/                        # Source Text/Manifests ([See Source Spec](Content/Content.md))
├── Data/                           # Staging Area for Raw Extraction ([See Data Spec](Data/Data.md))
├── Documentation/                  # Detailed Technical Docs per Module
├── Finalized/                      # PRODUCTION READY DATA ([See Finalized Spec](Finalized/Finalized.md))
│   ├── classified_images_final_unlabeled/ # Images sorted by class (for verify)
│   ├── Images/                     # The final set of "Keep" images
│   └── MD/                         # Cleaned Markdown files
├── Reference_Images/               # Ground Truth feature bank
└── Results/                        # Accuracy Reports ([See Results Spec](Results/Results.md))
```

## Tech Stack
*   **Python 3.10+**
*   **PyMuPDF (fitz)**: Document Parsing.
*   **OpenCV (cv2) & PIL**: Image Processing & Feature Extraction.
*   **Pandas/NumPy**: Statistical Analysis.

## Execution Guide

To reproduce the pipeline results, execute the modules in the following order:

### 1. Setup Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

```bash
pip install -r requirements.txt
```

### 2. Run Pipeline
1.  **Ingestion**: Extract content from raw PDFs.
    ```bash
    python Codes/PDF2MD.py
    ```
2.  *(Optional)* **Build Reference Set**: Manually curate `Reference_Images/` with 5-10 examples per class (Logo, Icon, etc.).
3.  **Classification**: Run the classifier to sort extracted images.
    ```bash
    python Codes/Clean_Image.py
    ```
4.  *(Optional)* **Validation**: Generate accuracy reports (requires labeled data).
    ```bash
    python Codes/Reports.py
    ```
5.  **Finalization**: Delete irrelevant images and update Markdown references.
    ```bash
    python Codes/cleanup_classified_images.py
    ```

## Usage Notice
*   **Accuracy**: The classification performance is directly checking the quality of the `Reference_Images` provided. If accuracy drops, adjust the reference set to better represent the target domain.
*   **File Paths**: Ensure strict directory structure is maintained as the scripts rely on relative path traversal.

## Research & Benchmarking
To define the operational boundaries of this tool, we conducted stress tests using the [Animal Image Dataset](Analysis/Analysis.md) (90 categories, 5400 images).

*   **The Challenge**: Distinguishing semantic concepts (e.g., *Wolf* vs. *Dog*) is difficult for rule-based systems that rely on low-level features (Color/Texture) rather than semantic understanding.
*   **Results**: The classifier performed poorly (~14% accuracy) on this semantic task, contrasting sharply with its high performance (>94%) on document layout tasks.
*   **Key Insight**: This confirms the tool's specialization: it is unbeatable for **Structure/Layout Classification** (PDFs) but should not be used for **Semantic Object Recognition** (Real-world photos).
*   **Full Analysis**: [Read the Research Report](Analysis/Analysis.md).

## License
This project is licensed under the **GNU Affero General Public License v3.0 (AGPLv3)**.

*   **Legal Constraint**: This project links against `PyMuPDF` (AGPLv3). Therefore, this codebase is strictly bound to the AGPLv3 license terms.
*   **SaaS Requirement**: If you run this software as a backend service (e.g., a "PDF Cleaning API") accessible over a network, you **must** provide the source code to any user interacting with that service.
*   **Commercial Use**: For proprietary/closed-source applications, you must acquire a valid commercial license for `PyMuPDF` from Artifex Software.

---
## Acknowledgements
*   **PyMuPDF Team**: For their exceptional PDF parsing library. 
*   **OpenCV Community**: For their powerful computer vision tools.
*   **Pandas/NumPy Developers**: For enabling efficient data manipulation and analysis.
*   **Kaggle**: For providing diverse datasets that facilitated our benchmarking experiments.
---
## Contact
For questions, suggestions, or contributions, please open an issue or submit a pull request on the GitHub repository.
*   **GitHub**: [https://github.com/Nikhil-SG](https://github.com/Nikhil-SG)
