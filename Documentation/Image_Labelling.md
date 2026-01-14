# Image_Labelling.py - Dataset Normalization Utility

## 1. Overview
`Image_Labelling.py` is a support utility designed for the rigorous management of image datasets. In the development of the classification logic, a "Golden Set" or "Ground Truth" dataset is required for validation. This script facilitates the creation of such datasets by normalizing filenames.

## 2. Problem Statement
When aggregating images from multiple PDF sources, filename collisions are frequent (e.g., multiple files named `page1_img1.jpg`). Furthermore, for automated reporting tools to verify accuracy, the "True Label" of an image must be retrievable.

## 3. Solution Algorithm
This script implements a two-pass renaming strategy to ensure data integrity and traceability:

1.  **Collision Avoidance**: 
    *   All files in the target directory are first renamed to a temporary Universally Unique Identifier (UUID). This prevents overwrite errors when sequence numbers overlap.
    *   Example: `img1.jpg` -> `.tmp_7f8a9d...jpg`

2.  **Semantic Serialization**:
    *   Files are renamed based on their parent directory, which serves as the "Ground Truth" label.
    *   Format: `[Category]_[Index].[Extension]`
    *   Example: `classified_images/Building/.tmp_7f8...jpg` -> `classified_images/Building/Building_042.jpg`

## 4. Usage Context
This tool is primarily used by developers or data curators when:
*   Preparing a new `Reference_Images` set.
*   Creating a specific test batch for the `Reports.py` validation module.
*   Merging new data into an existing training repository.

*Note: This is a destructive operation regarding filenames. It should only be used on datasets where the original filename (page reference) is no longer critical, or on copies of the data.*
