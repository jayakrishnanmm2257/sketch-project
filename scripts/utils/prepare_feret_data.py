import os
import bz2
from PIL import Image
from tqdm import tqdm
import io

# --- CONFIGURATION ---
# IMPORTANT: Update this path
FERET_IMAGES_DIR = r'C:\Users\krish\Downloads\Sketch Project\data\colorferet\dvd1\data\smaller' # The folder with subfolders like '00739'
CLEAN_OUTPUT_DIR = 'feret_frontal_photos_jpg_dvd1_smaller' # A new folder to save the final JPG photos

# The file ending for the primary frontal photo.
FRONTAL_POSE_CODE = '_fa.ppm.bz2'

print("Starting to process and convert the FERET database...")
os.makedirs(CLEAN_OUTPUT_DIR, exist_ok=True)

# Get a list of all the subject directories (e.g., '00739', '00740')
subject_folders = [f for f in os.listdir(FERET_IMAGES_DIR) if os.path.isdir(os.path.join(FERET_IMAGES_DIR, f))]

processed_files = 0
for subject_id in tqdm(subject_folders, desc="Processing Subjects"):
    subject_folder_path = os.path.join(FERET_IMAGES_DIR, subject_id)
    
    # Find the target frontal photo in the subject's folder
    target_filename = None
    for filename in os.listdir(subject_folder_path):
        if filename.endswith(FRONTAL_POSE_CODE):
            target_filename = filename
            break # Found it, no need to look further

    if target_filename:
        source_path = os.path.join(subject_folder_path, target_filename)
        
        # Create the new, simple JPG filename (e.g., '00739.jpg')
        new_filename_jpg = f"{subject_id}.jpg"
        destination_path = os.path.join(CLEAN_OUTPUT_DIR, new_filename_jpg)

        try:
            # Open the compressed bz2 file in binary read mode
            with open(source_path, 'rb') as bz2_file:
                # Decompress the data in memory
                ppm_data = bz2.decompress(bz2_file.read())
            
            # Use io.BytesIO to treat the in-memory data like a file
            # and open it with the Pillow library
            with Image.open(io.BytesIO(ppm_data)) as im:
                # Convert to RGB (standard for color photos) and save as a JPG
                im.convert('RGB').save(destination_path, 'jpeg')
            
            processed_files += 1

        except Exception as e:
            print(f"Error processing {target_filename}: {e}")

print(f"\nProcessing complete. Created {processed_files} clean JPG photos in '{CLEAN_OUTPUT_DIR}'")