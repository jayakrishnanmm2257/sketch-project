import os
import bz2
from PIL import Image, ImageOps
import io
import sys
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import config

# --- Configuration ---
SKETCH_FILENAMES_PATH = os.path.join(config.PROJECT_ROOT, 'data/CUHK/sketch_filenames.txt')
GRAYFERET_BASE_PATHS = [
    os.path.join(config.PROJECT_ROOT, 'data/colorferet/dvd2/gray_feret_cd1/data/images'),
    os.path.join(config.PROJECT_ROOT, 'data/colorferet/dvd2/gray_feret_cd2/data/images')
]
ORIGINAL_SKETCH_DIR = os.path.join(config.PROJECT_ROOT, 'data/CUHK/original_sketch')

# Output directories from config
OUTPUT_PHOTO_DIR = config.MATCHED_PHOTOS_DIR
OUTPUT_SKETCH_DIR = config.MATCHED_SKETCHES_DIR
OUTPUT_IMAGE_SIZE = (config.IMAGE_SIZE, config.IMAGE_SIZE)

def find_gray_feret_path(base_filename, base_paths):
    for base_path in base_paths:
        if not os.path.isdir(base_path):
            continue
        tif_bz2_path = os.path.join(base_path, f"{base_filename}.tif.bz2")
        if os.path.exists(tif_bz2_path):
            return tif_bz2_path
    return None

def crop_center(img, target_size=256):
    """
    Center crops the image to target_size x target_size.
    If image is smaller than target, it resizes the *smaller* edge to target_size first,
    preserving aspect ratio.
    """
    w, h = img.size
    
    # Resize if needed so that the *smallest* dimension is at least target_size
    min_dim = min(w, h)
    if min_dim < target_size:
        scale = target_size / min_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        w, h = img.size
    
    # Calculate crop coordinates
    left = (w - target_size) // 2
    top = (h - target_size) // 2
    right = left + target_size
    bottom = top + target_size
    
    return img.crop((left, top, right, bottom))

def process_sketch(img, target_size=256):
    """
    1. Crops to ink content.
    2. Resizes so the smaller dimension is target_size (usually width).
    3. Center crops to target_size x target_size.
    """
    # 1. Crop to Content
    inverted = ImageOps.invert(img)
    bbox = inverted.getbbox()
    if bbox:
        img = img.crop(bbox)
    
    # 2. Resize maintaining aspect ratio so shortest side = target_size
    w, h = img.size
    if w < h:
        # Portrait: Width becomes target_size
        new_w = target_size
        new_h = int(h * (target_size / w))
    else:
        # Landscape: Height becomes target_size
        new_h = target_size
        new_w = int(w * (target_size / h))
        
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 3. Center Crop to square
    return crop_center(img, target_size)

def prepare_original_dataset():
    print("--- Starting Dataset Preparation (using GrayFERET) ---")
    
    os.makedirs(OUTPUT_PHOTO_DIR, exist_ok=True)
    os.makedirs(OUTPUT_SKETCH_DIR, exist_ok=True)

    if not os.path.exists(SKETCH_FILENAMES_PATH):
        print(f"Error: {SKETCH_FILENAMES_PATH} not found.")
        return

    with open(SKETCH_FILENAMES_PATH, 'r') as f:
        sketch_filenames = [line.strip() for line in f.readlines() if line.strip()]

    processed_pairs = 0
    skipped_count = 0

    for sketch_filename in tqdm(sketch_filenames, desc="Matching & Resizing"):
        base_name = os.path.splitext(sketch_filename)[0]
        subject_id = sketch_filename[:5]
        
        sketch_path = os.path.join(ORIGINAL_SKETCH_DIR, f"{subject_id}.jpg")
        photo_path = find_gray_feret_path(base_name, GRAYFERET_BASE_PATHS)

        if not photo_path or not os.path.exists(sketch_path):
            skipped_count += 1
            continue

        try:
            # --- Process Photo ---
            with open(photo_path, 'rb') as bz2_file:
                tif_data = bz2.decompress(bz2_file.read())
            with Image.open(io.BytesIO(tif_data)).convert("RGB") as img:
                # Photos are 256x384. We want 256x256.
                # Just center cropping works perfectly for these.
                final_img = crop_center(img, target_size=OUTPUT_IMAGE_SIZE[0])
                output_filename = f"{base_name}.jpg"
                final_img.save(os.path.join(OUTPUT_PHOTO_DIR, output_filename), 'jpeg')

            # --- Process Sketch ---
            with Image.open(sketch_path) as sketch_img:
                sketch_img = sketch_img.convert("L")
                final_sketch = process_sketch(sketch_img, target_size=OUTPUT_IMAGE_SIZE[0])
                final_sketch.save(os.path.join(OUTPUT_SKETCH_DIR, output_filename), 'jpeg')

            processed_pairs += 1
        except Exception as e:
            print(f"Error processing {sketch_filename}: {e}")
            skipped_count += 1
            
    print(f"\nCompleted: {processed_pairs} pairs. Skipped: {skipped_count}.")

if __name__ == "__main__":
    prepare_original_dataset()