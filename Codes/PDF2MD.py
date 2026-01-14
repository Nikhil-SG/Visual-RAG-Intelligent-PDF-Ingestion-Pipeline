"""
PDF to Markdown Conversion Module

This module serves as the data ingestion layer for the RAG pipeline.
It handles the extraction of text and images from PDF documents, preserving the
logical flow and structure.

The extracted content is saved as:
1. Markdown files (.md) containing text and image references.
2. Image files (extracted from the PDF binary streams) in a separate directory.

Dependencies:
    - fitz (PyMuPDF): For low-level PDF parsing and image extraction.
    - pymupdf4llm: For layout-preserving text-to-markdown conversion.
"""

import os
import fitz  # PyMuPDF
import pymupdf4llm
from tqdm import tqdm
import pathlib
import time
import gc

def convert_pdf_to_markdown(pdf_path, output_md_path, images_dir):
    """
    Convert a PDF to markdown with images extracted and referenced.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        tqdm.write(f"Error opening {pdf_path}: {e}")
        return 0
    
    page_count = len(doc)
    full_markdown = ""
    total_images = 0
    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
    
    for page_num in range(page_count):
        page = doc[page_num]
        page_elements = []
        
        # Extract text blocks with their positions
        text_blocks = page.get_text("blocks")
        for block in text_blocks:
            if block[4].strip():
                page_elements.append({
                    "type": "text",
                    "rect": fitz.Rect(block[:4]),  # x0, y0, x1, y1
                    "content": block[4],
                    "y_pos": block[1]  # y0 coordinate
                })
        
        # Extract images with their positions
        image_list = page.get_images(full=True)
        total_images += len(image_list)
        
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            
            try:
                # Extract image
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Get image extension
                ext = base_image["ext"]
                if ext == "jpeg":
                    ext = "jpg"
                
                # Create image filename
                image_filename = f"{pdf_basename}_p{page_num+1}_img{img_index+1}.{ext}"
                image_path = os.path.join(images_dir, image_filename)
                
                # Save image
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                # Find image rectangles on the page
                img_rects = page.get_image_rects(xref)
                
                if img_rects:
                    img_rect = img_rects[0]
                    page_elements.append({
                        "type": "image",
                        "rect": img_rect,
                        "filename": image_filename,
                        "y_pos": img_rect.y0  # y0 coordinate
                    })
            
            except Exception as e:
                tqdm.write(f"Warning: Could not extract image {img_index} on page {page_num+1}: {e}")
        
        # Sort all elements by their y-position
        page_elements.sort(key=lambda x: x["y_pos"])
        
        # Create a temporary PDF file with just this page for text extraction
        temp_pdf_path = f"temp_page_{page_num+1}.pdf"
        temp_doc = fitz.open()
        temp_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
        temp_doc.save(temp_pdf_path)
        temp_doc.close()
        
        try:
            # Get the raw markdown text for this page
            raw_md_text = pymupdf4llm.to_markdown(temp_pdf_path)
            
            if len(page_elements) > 0:
                # Reconstruct the markdown with images inserted at correct positions
                page_markdown = ""
                text_blocks_only = [el for el in page_elements if el["type"] == "text"]
                
                if len(text_blocks_only) == 0:
                    # Page with only images
                    for el in page_elements:
                        if el["type"] == "image":
                            page_markdown += f"\n![IMAGE_REF]({el['filename']})\n"
                else:
                    # Use pymupdf4llm's markdown as the base text content
                    md_text = raw_md_text
                    md_chunks = md_text.split('\n\n')
                    
                    if len(md_chunks) < len(text_blocks_only) / 2:
                        md_chunks = md_text.split('\n')
                    
                    current_chunk = 0
                    
                    for el in page_elements:
                        if el["type"] == "text":
                            if current_chunk < len(md_chunks):
                                page_markdown += md_chunks[current_chunk] + "\n"
                                current_chunk += 1
                        else:  # image
                            page_markdown += f"\n![IMAGE_REF]({el['filename']})\n"
                    
                    # Add any remaining chunks
                    while current_chunk < len(md_chunks):
                        page_markdown += md_chunks[current_chunk] + "\n"
                        current_chunk += 1
            else:
                page_markdown = raw_md_text
            
            full_markdown += page_markdown
        
        except Exception as e:
            tqdm.write(f"Error converting page {page_num+1} to markdown: {e}")
            full_markdown += f"[Error converting page {page_num+1}]\n\n"
        
        finally:
            gc.collect()
            time.sleep(0.1)
            try:
                os.remove(temp_pdf_path)
            except PermissionError:
                tqdm.write(f"Warning: Could not remove {temp_pdf_path}, will try later")
    
    # Close document
    doc.close()
    
    # Clean up any remaining temporary files
    for i in range(page_count):
        temp_pdf_path = f"temp_page_{i+1}.pdf"
        if os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
            except:
                pass
    
    # Save markdown file
    pathlib.Path(output_md_path).write_text(full_markdown, encoding="utf-8")
    
    return total_images


def batch_convert_pdfs_to_markdown(root_folder):
    """
    Recursively find all PDFs in root_folder and convert each to markdown.
    Save all markdown files in a 'MD' folder and all images in an 'Images' folder.
    """
    # Use specified base output directory (change as needed)
    # Try using the configured path first; if it's not writable, fall back
    # to a safe directory under the current user's home.
    configured_base = pathlib.Path(r"E:\Master's\2nd Year\MyWork\Data")
    fallback_base = pathlib.Path.home() / "PDF2MD_Output"

    base_output_dir = configured_base
    try:
        # Ensure we can create and write to the configured path
        base_output_dir.mkdir(parents=True, exist_ok=True)
        test_file = base_output_dir / ".perm_test"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("perm test")
        test_file.unlink()
    except Exception:
        print(f"Warning: cannot create or write to {configured_base!s}. Falling back to {fallback_base!s}")
        base_output_dir = fallback_base

    data_folder = str(base_output_dir / "MD")
    images_folder = str(base_output_dir / "Images")

    # Create output directories
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)
    
    print(f"Created output folders:")
    print(f"  Markdown files: {data_folder}")
    print(f"  Images: {images_folder}\n")
    
    # Collect all PDF files first
    pdf_files = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(dirpath, filename))
    
    print(f"Found {len(pdf_files)} PDF files to process\n")
    
    # Process with progress bar
    for pdf_path in tqdm(pdf_files, desc="Converting PDFs", unit="file"):
        pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
        md_path = os.path.join(data_folder, pdf_basename + ".md")
        
        total_images = convert_pdf_to_markdown(pdf_path, md_path, images_folder)
        tqdm.write(f"  {pdf_basename}: {total_images} images extracted")
    
    print("\n" + ("-" * 50))
    print(f"Conversion complete! Processed {len(pdf_files)} PDF files.")
    print("-" * 50)


if __name__ == "__main__":
    # SET THIS LINE
    input_dir = r"E:\Master's\2nd Year\MyWork\Content"
    batch_convert_pdfs_to_markdown(input_dir)