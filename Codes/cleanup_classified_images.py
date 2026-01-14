"""
Post-Classification Cleanup Utility

This script finalizes the dataset by reconciling the file system with the generated markdown files.
It performs two primary functions based on the classification results:
1. Deletes images that were classified into non-relevant categories (e.g., removing 'Logos' or 'Icons' from the main source).
2. Sanitizes the associated Markdown documents by removing image reference tags pointing to deleted files.

This ensures the final dataset contains only relevant visual associations.
"""

import os
import re

# Directory paths
# Define paths relative to the script location (assuming script is in Codes/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_COPY_DIR = os.path.join(BASE_DIR, "Finalized", "Images")
DATA_COPY_DIR = os.path.join(BASE_DIR, "Finalized", "MD")
CLASSIFIED_COPY_DIR = os.path.join(BASE_DIR, "Finalized", "classified_images")

def get_md_filename(image_name):
    """Remove page no, image no and extension from image name to get markdown filename"""
    # Pattern matches: _p<page_number>_img<image_number>.<extension>
    # Example: "2024 Prospectus_p260_img1.jpg" -> "2024 Prospectus"
    base = re.sub(r'_p\d+_img\d+\.[a-zA-Z]+$', '', image_name)
    return f"{base}.md"

def remove_image_ref_from_md(md_path, image_name):
    """Remove image reference lines from markdown file"""
    if not os.path.isfile(md_path):
        print(f"⚠ Markdown file not found: {md_path}")
        return
    
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Pattern to match lines containing IMAGE_REF with the image name
    pattern = f".*IMAGE_REF.*{re.escape(image_name)}.*"
    new_lines = []
    removed_count = 0
    
    for line in lines:
        if re.search(pattern, line):
            new_lines.append("\n")  # Replace with single newline
            removed_count += 1
        else:
            new_lines.append(line)
    
    if removed_count > 0:
        with open(md_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        print(f"  ✓ Removed {removed_count} reference(s) from: {md_path}")

def get_all_images_in_subfolders(folder):
    """Get all image files in folder and its subfolders"""
    image_files = set()
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_files.add(file)
    return image_files

def main():
    """Main function to cleanup classified images and their markdown references"""
    print("=" * 80)
    print("CLASSIFIED IMAGES CLEANUP SCRIPT")
    print("=" * 80)
    
    # Verify directories exist
    if not os.path.exists(CLASSIFIED_COPY_DIR):
        print(f"❌ Error: Classified directory not found: {CLASSIFIED_COPY_DIR}")
        return
    
    if not os.path.exists(IMAGES_COPY_DIR):
        print(f"❌ Error: Images directory not found: {IMAGES_COPY_DIR}")
        return
    
    if not os.path.exists(DATA_COPY_DIR):
        print(f"❌ Error: Data directory not found: {DATA_COPY_DIR}")
        return
    
    # Get all classified images
    print(f"\n📂 Scanning classified images in: {CLASSIFIED_COPY_DIR}")
    classified_images = get_all_images_in_subfolders(CLASSIFIED_COPY_DIR)
    print(f"   Found {len(classified_images)} classified images")
    
    # Get all images in the images copy directory
    print(f"\n📂 Scanning images in: {IMAGES_COPY_DIR}")
    images_copy = set([f for f in os.listdir(IMAGES_COPY_DIR) 
                       if os.path.isfile(os.path.join(IMAGES_COPY_DIR, f))])
    print(f"   Found {len(images_copy)} images")
    
    # Find common images (images that are classified and still in images copy)
    common_images = classified_images & images_copy
    
    print("\n" + "=" * 80)
    print(f"Found {len(common_images)} images to delete and update references")
    print("=" * 80)
    
    if len(common_images) == 0:
        print("✓ No images to cleanup. All done!")
        return
    
    deleted_count = 0
    updated_count = 0
    
    for i, image_name in enumerate(common_images, 1):
        print(f"\n[{i}/{len(common_images)}] Processing: {image_name}")
        
        # Delete image from Images Copy
        img_path = os.path.join(IMAGES_COPY_DIR, image_name)
        if os.path.isfile(img_path):
            try:
                os.remove(img_path)
                print(f"  ✓ Deleted: {img_path}")
                deleted_count += 1
            except Exception as e:
                print(f"  ❌ Error deleting {img_path}: {e}")
        
        # Find corresponding markdown file in Data Copy
        md_filename = get_md_filename(image_name)
        md_path = os.path.join(DATA_COPY_DIR, md_filename)
        
        if os.path.isfile(md_path):
            remove_image_ref_from_md(md_path, image_name)
            updated_count += 1
        else:
            print(f"  ⚠ Markdown file not found: {md_path}")
    
    print("\n" + "=" * 80)
    print("CLEANUP COMPLETE!")
    print("=" * 80)
    print(f"✓ Deleted {deleted_count} images")
    print(f"✓ Updated {updated_count} markdown files")
    print("=" * 80)

if __name__ == "__main__":
    main()
