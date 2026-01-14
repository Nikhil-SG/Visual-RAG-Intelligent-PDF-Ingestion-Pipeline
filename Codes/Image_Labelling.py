"""
Image Labelling and Normalization Utility

This utility standardizes image filenames within the dataset.
It addresses the issue of filename collisions when merging extracted images from different sources
and prepares the dataset for evaluation or reporting.

Algorithm:
1. Temporarily renaming files to UUIDs to prevent collision during processing.
2. Sequentially renaming files based on their parent directory name (Ground Truth Label).
   Format: {DirectoryName}_{Index}.{Extension}
"""

import os, uuid

# Define paths relative to the script location (assuming script is in Codes/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.join(BASE_DIR, "Data", "classified_images")

def rename_subfolders(parent):
    parent = os.path.normpath(parent)
    for sub in sorted(os.listdir(parent)):
        subpath = os.path.join(parent, sub)
        if not os.path.isdir(subpath):
            continue
        files = [f for f in sorted(os.listdir(subpath)) if os.path.isfile(os.path.join(subpath, f))]
        if not files:
            continue
        temps = []
        for f in files:
            old = os.path.join(subpath, f)
            ext = os.path.splitext(f)[1]
            temp = os.path.join(subpath, ".tmp_" + uuid.uuid4().hex + ext)
            os.rename(old, temp)
            temps.append((temp, ext))
        for i, (temp, ext) in enumerate(temps, 1):
            new = os.path.join(subpath, f"{sub}_{i}{ext}")
            os.rename(temp, new)

if __name__ == "__main__":
    rename_subfolders(PARENT)
    print("Done")