import os
import bz2
from PIL import Image
import io
import numpy as np
from tqdm import tqdm

# --- Configuration ---
SKETCH_FILENAMES_PATH = 'data/CUHK/sketch_filenames.txt'
COLORFERET_BASE_PATHS = [
    'data/colorferet/dvd1/data/images',
    'data/colorferet/dvd2/data/images',
    'data/colorferet/dvd1/data/smaller',
    'data/colorferet/dvd2/data/smaller'
]
ORIGINAL_SKETCH_DIR = 'data/CUHK/original_sketch'

OUTPUT_PHOTO_DIR = 'data/matched_photos_original'
OUTPUT_SKETCH_DIR = 'data/matched_sketches_original'
OUTPUT_IMAGE_SIZE = (256, 256)

def find_feret_file_path(subject_id, date_code, pose_code, subject_to_path_map):
    if subject_id not in subject_to_path_map:
        return None
    subject_dir = os.path.join(subject_to_path_map[subject_id], subject_id)
    if os.path.isdir(subject_dir):
        for f in os.listdir(subject_dir):
            if f.startswith(f"{subject_id}_{date_code}_{pose_code}"):
                return os.path.join(subject_dir, f)
    return None

def prepare_original_dataset():
    """
    Pairs original, uncropped photos with original, uncropped sketches.
    Resizes both to a standard square dimension without complex cropping.
    """
    print("--- Starting Dataset Preparation (Original Images) ---")

    os.makedirs(OUTPUT_PHOTO_DIR, exist_ok=True)
    os.makedirs(OUTPUT_SKETCH_DIR, exist_ok=True)
    print(f"Output directories '{OUTPUT_PHOTO_DIR}' and '{OUTPUT_SKETCH_DIR}' are ready.")

    subject_to_path_map = {}
    print("Building map of subject locations...")
    for base_path in COLORFERET_BASE_PATHS:
        if os.path.isdir(base_path):
            for subject_id in os.listdir(base_path):
                if os.path.isdir(os.path.join(base_path, subject_id)) and subject_id not in subject_to_path_map:
                    subject_to_path_map[subject_id] = base_path
    
    with open(SKETCH_FILENAMES_PATH, 'r') as f:
        sketch_filenames = [line.strip() for line in f.readlines() if line.strip()]

    processed_pairs, skipped_count = 0, 0

    for sketch_filename in tqdm(sketch_filenames, desc="Processing Pairs"):
        base_name, subject_id = os.path.splitext(sketch_filename)[0], sketch_filename[:5]
        
        sketch_path = os.path.join(ORIGINAL_SKETCH_DIR, f"{subject_id}.jpg")
        date_code, pose_code = sketch_filename.split('_')[-1].split('.')[0], sketch_filename[5:7]
        photo_path = find_feret_file_path(subject_id, date_code, pose_code, subject_to_path_map)

        if not all([os.path.isfile(sketch_path), photo_path]):
            skipped_count += 1
            continue

        try:
            # Process Photo
            with open(photo_path, 'rb') as bz2_file:
                ppm_data = bz2.decompress(bz2_file.read())
            with Image.open(io.BytesIO(ppm_data)).convert("RGB") as img:
                resized_img = img.resize(OUTPUT_IMAGE_SIZE, Image.Resampling.LANCZOS)
                output_filename = f"{base_name}.jpg"
                resized_img.save(os.path.join(OUTPUT_PHOTO_DIR, output_filename), 'jpeg')

            # Process Sketch
            with Image.open(sketch_path) as sketch_img:
                resized_sketch = sketch_img.resize(OUTPUT_IMAGE_SIZE, Image.Resampling.LANCZOS)
                resized_sketch.save(os.path.join(OUTPUT_SKETCH_DIR, output_filename), 'jpeg')

            processed_pairs += 1
        except Exception:
            skipped_count += 1
            
    print("\n--- Dataset Preparation Report ---")
    print(f"✅ Successfully processed and saved {processed_pairs} original photo-sketch pairs.")
    print(f"❌ Skipped (missing component or error): {skipped_count}")
    print("------------------------------------")

if __name__ == "__main__":
    prepare_original_dataset()
