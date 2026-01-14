# PDF2MD.py - PDF to Markdown Extraction Module

## 1. Overview
The `PDF2MD.py` module serves as the initial ingestion layer of the RAG (Retrieval-Augmented Generation) pipeline. Its primary objective is to convert unstructured binary PDF documents into a structured, machine-readable Markdown format while preserving the semantic flow of information.

Crucially, this module addresses a common deficiency in text-only extractors by treating the PDF as a multimodal document. It captures valid visual assets (charts, diagrams, photos) along with the text, ensuring that the resulting Markdown file faithfully represents the original document's layout and content.

## 2. RAG & Token Optimization Context
In standard RAG pipelines, ingesting raw PDF content often leads to identifying everything as "content." This results in:
1.  **Context Pollution**: Headers, footers, and decorative icons are extracted as valid images or text.
2.  **Token Watage**: Large numbers of irrelevant image references fill up the context window of Multimodal LLMs, reducing the model's ability to focus on pertinent data.

This module mitigates this by extracting *all* media first, maintaining their specific insertion points in the text flow. This "grab-everything-preserving-order" approach allows downstream modules (`Clean_Image.py`) to selectively filter visual noise while keeping the textual narrative intact.

## 3. Technical Implementation
The script utilizes **PyMuPDF (fitz)** and **pymupdf4llm** for high-fidelity parsing.

### Core Workflow:
1.  **Iterative Parsing**: Processes the document page-by-page to minimize memory footprint.
2.  **Text Block Analysis**: Identifies text coordinates (rectangles) to reconstruct the reading order.
3.  **Binary Image Extraction**:
    *   Extracts images from the PDF object stream.
    *   Converts formats (e.g., JPEG, PNG) to standard compatible formats.
    *   Saves images to a dedicated `Images/` directory with a naming convention linking them to the source page (e.g., `DocName_p1_img1.jpg`).
4.  **Markdown reconstruction**:
    *   Unlike simple OCR, this method preserves headers, bold text, and lists.
    *   Injects markdown image tags (`![Image](path)`) at the exact vertical position (`y_pos`) where the image appeared in the original PDF.

## 4. Usage & Dependencies
This script is optimized for handling complex layouts found in brochures and prospectuses.

*   **Input**: Raw PDF files.
*   **Output**: 
    *   Markdown (.md) files in the `Data/MD` directory.
    *   Extracted assets in the `Data/Images` directory.
*   **Key Libraries**: `fitz` (PyMuPDF), `pymupdf4llm`.
