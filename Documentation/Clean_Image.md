# Clean_Image.py - Rule-Based Image Classification System

## 1. Overview
`Clean_Image.py` functions as the filtering engine of the pipeline. Following the raw extraction phase, the dataset typically contains a mixture of high-value "Signal" images (graphs, photos) and low-value "Noise" (icons, vectors, logos).

This module prevents the "Garbage In, Garbage Out" phenomenon in RAG systems by segregating these image types. It employs a **Reference-Based Computer Vision Classifier** rather than a "Black Box" Neural Network. This ensures explainability, speed, and privacy, as processing occurs locally without external API calls.

## 2. Methodology: Human-in-the-Loop Reference System
Instead of training a model from scratch, this system uses a "Reference Set" to define categories. 
1.  **Manual Curation**: A user manually creates folders (e.g., `Reference_Images/Icon`, `Reference_Images/Content`) and places 5-10 representative images in each.
2.  **Dynamic Learning**: The script scans these folders at runtime to "learn" the categories.
3.  **Feature Matching**: New, unclassified images are compared against these references.

This approach is highly adaptable; to detect a new class of "Noise" (e.g., Watermarks), one simply adds a `Watermarks` folder to the references, effectively "programming" the classifier with examples.

## 3. Technical Implementation
The `AdvancedImageClassifier` class implements a weighted similarity algorithm based on classical computer vision features.

### Feature Extraction Vector:
*   **Geometry**: Aspect Ratio, Area (Logarithmic), Layout orientation.
*   **Color Analysis**: Mean RGB values, Brightness, Contrast, and Color Histograms.
*   **Texture & Structure**: Laplacian Variance (sharpness), Corner Density (Harris Corners), and Edge Density (Canny).

### Algorithm:
1.  **Profile Generation**: Calculates the feature vector for every image in the `Reference_Images` folder.
2.  **Comparison**: For every new image, calculates the Euclidean distance and Histogram Correlation against the reference profiles.
3.  **Voting**: The category with the highest aggregate similarity score is assigned to the image.

## 4. Usage
*   **Input**: Unsorted images from the `PDF2MD` extraction.
*   **Configuration**: Requires the `Reference_Images` directory to be populated.
*   **Output**: Moves images into structured subdirectories within `classified_images/` (e.g., `classified_images/Logo/`).

*Note: The accuracy of this module is directly dependent on the quality of the Reference Images provided.*
