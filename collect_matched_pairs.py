import os
import shutil
from PIL import Image
import bz2

# --- Configuration ---
# Path to the file containing the list of original FERET filenames that were sketched.
SKETCH_FILENAMES_LIST = 'data/CUHK/sketch_filenames.txt'

# Directory containing the original ColorFERET images (in .ppm.bz2 format).
# The structure is assumed to be /dvd1/data/images/YYYYY/
COLORFERET_BASE_DIR = 'data/colorferet/dvd1/data/images' 

# Directory containing the CUHK cropped sketches.
CROPPED_SKETCH_DIR = 'data/CUHK/cropped_sketch'

# Directory containing the CUHK photo landmark points (.3pts files).
PHOTO_POINTS_DIR = 'data/CUHK/photo_points'

# Output directories for the matched pairs.
MATCHED_PHOTO_DIR = 'data/matched_photos'
MATCHED_SKETCH_DIR = 'data/matched_sketches'

# Number of images to process for this test run.
IMAGE_LIMIT = 5

# --- Main Script ---

def create_dir_if_not_exists(directory):
    """Creates a directory if it doesn't already exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def find_feret_photo_path(base_dir, feret_filename_from_list):
    """
    Finds the full path to a ColorFERET photo given an entry from sketch_filenames.txt.
    E.g., '00001fb010_930831.jpg' -> 'data/colorferet/dvd1/data/images/00001/00001_930831_fb_a.ppm.bz2'
    """
    # Remove .jpg extension to get base name (e.g., '00001fb010_930831')
    base_name_from_list = feret_filename_from_list.rsplit('.', 1)[0]
    
    # Extract subject ID (first 5 characters, e.g., '00001')
    subject_id = base_name_from_list[:5]
    
    # Extract date part (last 6 characters, e.g., '930831')
    date_part = base_name_from_list[-6:]
    
    # Extract view identifier (e.g., 'fb' from 'fb010')
    # This assumes 'fb010' means 'fb_a' or similar.
    # The part between subject_id (5 chars) and date_part (6 chars) is the view code + index
    view_code_raw = base_name_from_list[5:-7] # e.g., 'fb010'
    view_code_prefix = view_code_raw[:2] # e.g., 'fb'

    subject_dir = os.path.join(base_dir, subject_id)
    
    if not os.path.isdir(subject_dir):
        return None # Subject directory not found

    # Iterate through files in the subject's directory to find a match
    for filename_in_feret_dir in os.listdir(subject_dir):
        if filename_in_feret_dir.endswith(".ppm.bz2"):
            # Expected format: 00001_930831_fa_a.ppm.bz2
            # Check if filename starts with the correct subject ID and date
            if filename_in_feret_dir.startswith(f"{subject_id}_{date_part}"):
                # Check if it contains the view code (e.g., '_fb_a')
                # This is a heuristic match, assuming '010' maps to '_a' or similar common suffix
                if f"_{view_code_prefix}_a" in filename_in_feret_dir or \
                   f"_{view_code_prefix}_b" in filename_in_feret_dir or \
                   f"_{view_code_prefix}_c" in filename_in_feret_dir or \
                   f"_{view_code_prefix}_d" in filename_in_feret_dir:
                    return os.path.join(subject_dir, filename_in_feret_dir)

    return None # No matching file found

def crop_photo_using_landmarks(image, points_path):
    """
    Crops and resizes a photo based on facial landmark coordinates.
    The goal is to create a cropped photo that aligns with the cropped sketches.
    """
    # 1. Read the landmark points (eye1, eye2, mouth)
    with open(points_path, 'r') as f:
        points = [list(map(float, line.strip().split())) for line in f]
    
    eye1_x, eye1_y = points[0]
    eye2_x, eye2_y = points[1]
    
    # 2. Define target dimensions and aspect ratio
    TARGET_WIDTH = 200
    TARGET_HEIGHT = 250
    TARGET_ASPECT_RATIO = TARGET_WIDTH / TARGET_HEIGHT # 0.8

    # 3. Calculate crop box based on landmark heuristics
    # Eye center and width
    eye_center_x = (eye1_x + eye2_x) / 2
    eye_center_y = (eye1_y + eye2_y) / 2
    eye_width = abs(eye2_x - eye1_x)
    
    # Estimate face width based on eye width. A factor of 2.2 is a reasonable heuristic.
    face_width = eye_width * 2.2
    
    # Derive crop height from face_width to match the target aspect ratio
    # crop_height = face_width / TARGET_ASPECT_RATIO
    crop_height = face_width / 0.8 # (200/250)

    # Define the bounding box
    # The y-offset for the eye center is chosen to position the eyes in the upper part of the face.
    y_offset_ratio = 0.4 
    
    x_start = eye_center_x - (face_width / 2)
    y_start = eye_center_y - (crop_height * y_offset_ratio)
    
    x_end = x_start + face_width
    y_end = y_start + crop_height

    # 4. Crop the image using the calculated box
    crop_box = (int(x_start), int(y_start), int(x_end), int(y_end))
    cropped_image = image.crop(crop_box)
    
    # 5. Resize to the exact target dimensions to ensure perfect alignment
    final_image = cropped_image.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
    
    return final_image

def main():
    """
    Collects and matches a small subset of ColorFERET photos and CUHK sketches.
    """
    print("Starting dataset matching process...")
    
    # 1. Create output directories
    create_dir_if_not_exists(MATCHED_PHOTO_DIR)
    create_dir_if_not_exists(MATCHED_SKETCH_DIR)

    # 2. Read the list of sketched filenames
    try:
        with open(SKETCH_FILENAMES_LIST, 'r') as f:
            # Read lines, strip whitespace, and filter out any empty lines
            all_filenames = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"ERROR: Cannot find the crucial mapping file: {SKETCH_FILENAMES_LIST}")
        print("Please ensure the file exists and the path is correct.")
        return

    if not all_filenames:
        print("ERROR: The mapping file is empty. Cannot proceed.")
        return
        
    print(f"Found {len(all_filenames)} total entries in the mapping file.")
    print(f"Processing the first {IMAGE_LIMIT} images for verification...")

    processed_count = 0
    # 3. Loop through the limited list of filenames
    for raw_feret_filename_from_list in all_filenames[:IMAGE_LIMIT]:
        # Extract subject ID (first 5 chars) from the raw list entry
        subject_id = raw_feret_filename_from_list[:5]
        
        # --- Define file paths ---
        # Original Photo Path (compressed), found using the new robust function
        photo_path_bz2 = find_feret_photo_path(COLORFERET_BASE_DIR, raw_feret_filename_from_list)
        
        # Cropped Sketch Path (e.g., '00001.jpg')
        sketch_filename_jpg = f"{subject_id}.jpg"
        sketch_path_jpg = os.path.join(CROPPED_SKETCH_DIR, sketch_filename_jpg)
        
        # Destination Paths (always '00001.jpg' format for matching)
        dest_photo_path = os.path.join(MATCHED_PHOTO_DIR, f"{subject_id}.jpg")
        dest_sketch_path = os.path.join(MATCHED_SKETCH_DIR, f"{subject_id}.jpg")

        # --- Verification and Processing ---
        # Construct path to the corresponding landmark file
        points_filename = raw_feret_filename_from_list.replace('.jpg', '.3pts')
        points_path = os.path.join(PHOTO_POINTS_DIR, points_filename)

        if photo_path_bz2 is None:
            print(f"  [WARNING] Skipping {subject_id}: No matching ColorFERET photo found for {raw_feret_filename_from_list}")
            continue
            
        if not os.path.exists(sketch_path_jpg):
            print(f"  [WARNING] Skipping {subject_id}: Sketch not found at {sketch_path_jpg}")
            continue
            
        if not os.path.exists(points_path):
            print(f"  [WARNING] Skipping {subject_id}: Landmark file not found at {points_path}")
            continue

        print(f"  Processing pair for subject ID: {subject_id}")

        # 4. Process the Photo (Decompress, Crop, and Save)
        try:
            with bz2.open(photo_path_bz2, 'rb') as bz2f:
                with Image.open(bz2f) as img:
                    # Print the image size for debugging
                    print(f"    -> Original image size: {img.size}")
                    # Crop the image using the landmark points
                    cropped_img = crop_photo_using_landmarks(img, points_path)
                    # Convert to RGB (if not already) and save as JPEG
                    cropped_img.convert('RGB').save(dest_photo_path, 'jpeg')
        except Exception as e:
            print(f"    [ERROR] Failed to process photo {photo_path_bz2}: {e}")
            continue

        # 5. Copy the Sketch
        try:
            shutil.copy(sketch_path_jpg, dest_sketch_path)
        except Exception as e:
            print(f"    [ERROR] Failed to copy sketch {sketch_path_jpg}: {e}")
            continue
            
        processed_count += 1

    print("-" * 20)
    print(f"Verification complete. Processed {processed_count}/{IMAGE_LIMIT} pairs.")
    print(f"Check the output directories:")
    print(f"  Photos: {MATCHED_PHOTO_DIR}")
    print(f"  Sketches: {MATCHED_SKETCH_DIR}")


if __name__ == '__main__':
    main()
