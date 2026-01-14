# Production Data & Cleanup Protocol

## Overview
This directory contains the finalized, "Production-Ready" assets for the RAG pipeline. The content here represents the cleanest version of the data, where irrelevant visual noise has been meticulously removed to optimize token usage and retrieval quality.

## Directory Manifest
*   **`MD/`**: Sanitize Markdown files. All references to deleted images have been computationally removed, ensuring no broken links exist.
*   **`Images/`**: The curated image repository. It contains *only* the high-value images (Charts, Diagrams, Content) that survived the filtration process.
*   **`classified_images_final_unlabeled/`**: The staging area used to determine which files to delete.

---

## The Cleanup Workflow

The [cleanup_classified_images.py](../Codes/cleanup_classified_images.py) script operates destructively. To ensure data safety, follow this strict protocol:

### 1. Preparation (Manual Filter)
The classification script moves images into folders like `Icon`, `Logo`, `Content`.
*   **Action**: Go into the `classified_images` folder.
*   **Delete** the subfolders containing images you wish to **KEEP** (e.g., delete the `Content` and `Building` folders).
*   **Retain** only the subfolders containing images you wish to **DESTROY** (e.g., keep `Icon`, `Logo`, `Adverts`).

*The script interprets the presence of an image in this folder as a "Death Sentence" for that image.*

### 2. Execution
Run the cleanup script. It will:
1.  Scan the remaining folders (the "Junk" folders).
2.  Locate the corresponding original files in the `Images/` directory and permanently delete them.
3.  Parse the Markdown files in `MD/`, find the specific `![IMAGE_REF]` tags for those deleted images, and remove the lines.

### 3. CRITICAL: Naming Convention Warning
The cleanup automation relies heavily on file naming conventions to map images back to their parent Markdown files.

*   **Required Format**: `DocumentName_p{Page}_img{Index}.jpg` (Default output of extraction).
*   **The Risk**: If you have used [Image_Labelling.py](../Codes/Image_Labelling.py) to rename files to `Label_001.jpg`, the cleanup script **WILL FAIL**. It cannot deduce the parent markdown file from a generic label name.
*   **Requirement**: If you are using labeled images, you must maintain an external index mapping `Label_001.jpg` -> `Original_Name.jpg`. Without this, the system cannot perform the markdown cleaning.
*   **Best Practice**: Perform the cleanup on the *original extracted filenames* first. Only rename/label the images for machine learning training *after* the content pipeline is finalized.
