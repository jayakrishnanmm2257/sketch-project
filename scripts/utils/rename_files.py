import os

# --- IMPORTANT: Update these paths ---
PHOTOS_DIR = r'C:\Users\krish\Downloads\Sketch Project\data\CUHK\photos'
SKETCHES_DIR = r'C:\Users\krish\Downloads\Sketch Project\data\CUHK\sketches'

print("Starting renaming process based on file order...")

# Get the list of filenames from both directories
photo_files = os.listdir(PHOTOS_DIR)
sketch_files = os.listdir(SKETCHES_DIR)

# --- CRUCIAL STEP: Sort both lists alphabetically ---
# This ensures that photo_files[i] corresponds to sketch_files[i]
photo_files.sort()
sketch_files.sort()

# --- Sanity Check: Ensure both folders have the same number of files ---
if len(photo_files) != len(sketch_files):
    print("Error: The number of files in the photos and sketches directories do not match!")
    print(f"Photos count: {len(photo_files)}, Sketches count: {len(sketch_files)}")
    print("Cannot proceed with renaming. Please check your dataset.")
else:
    files_renamed = 0
    # Loop through the files by index
    for i in range(len(photo_files)):
        # The original sketch file we want to rename
        old_sketch_filename = sketch_files[i]
        
        # The new name we want to give it, which is the photo's filename
        new_sketch_filename = photo_files[i]
        
        # Get the full paths for the rename operation
        old_file_path = os.path.join(SKETCHES_DIR, old_sketch_filename)
        new_file_path = os.path.join(SKETCHES_DIR, new_sketch_filename)
        
        # Rename the file
        os.rename(old_file_path, new_file_path)
        files_renamed += 1
        
    print(f"Renaming complete. {files_renamed} files were renamed successfully.")