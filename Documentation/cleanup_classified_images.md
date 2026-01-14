# cleanup_classified_images.py - Dataset Synchronization & Sanitization

## 1. Overview
The `cleanup_classified_images.py` script serves as the finalization layer of the pipeline. Once images have been extracted and classified, a critical step remains: removing the "Noise" from the dataset and ensuring the textual content reflects these changes.

In a RAG context, simply deleting an image file is insufficient. If a Markdown file retains a reference `![Image](logo.png)` to a deleted file, it creates broken links and explicitly tells the LLM "there is an image here" when there is not. This script handles the synchronization between the file system and the document content.

## 2. Operational Logic
This script executes a destructive cleanup process based on the decisions made in the classification phase.

### Workflow:
1.  **Scanning**: It inventories the `Finalized/classified_images` directory. It assumes that images remaining in specific folders (like 'Logo' or 'Icon') are targeted for removal (or that the user has reviewed the folders and confirmed the selection).
2.  **redundancy check**: It identifies images that exist in both the `Images/` source folder and the classified folders.
3.  **File System Cleanup**: It physically deletes the targets from the main `Images/` directory, reducing storage size.
4.  **Markdown Sanitization**:
    *   It parses the corresponding Markdown files.
    *   It locates the specific `![IMAGE_REF]` tags corresponding to the deleted files.
    *   It excises these lines, leaving the surrounding text text intact.

## 3. Outcomes
*   **Token Efficiency**: The final Markdown files are stripped of references to non-informative visual elements.
*   **Data Integrity**: Broken links are prevented before the data is indexed by the vector database.
*   **Storage Optimization**: Duplicate and irrelevant image assets are permanently removed.

## 4. Usage
This script should be run **after** `Clean_Image.py` and after a manual review of the classified folders.

*   **Target Directory**: `Finalized/MD` and `Finalized/Images`.
*   **Pre-requisite**: Classification steps must be complete, and the classified folders must reflect the desired state of the data.
