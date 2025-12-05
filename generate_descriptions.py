import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm

# --- Function Definition ---
# Place the get_model function here, after the imports.
def get_model(num_attributes):
    # Initialize the ResNet34 architecture without pre-trained weights,
    # as we will be loading our own trained weights.
    model = resnet34(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_attributes)
    )
    return model

# --- CONFIGURATION ---
# IMPORTANT: Update these paths to your local folders
MODEL_PATH = 'facial_attribute_classifier.pth'
CUFS_PHOTOS_DIR = 'C:/Users/krish/Downloads/Sketch Project/data/CUHK/photos'
OUTPUT_CSV_PATH = 'sketch_descriptions.csv'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# --- 1. Load the Trained Model ---
# Now we call the get_model function we just defined
model = get_model(num_attributes=40)
# Load the weights, ensuring they are mapped to the correct device (CPU or GPU)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
model.to(DEVICE)
model.eval()  # IMPORTANT: Set model to evaluation mode

# --- 2. Define Image Transformations ---
# These MUST be identical to the transformations used during training
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 3. List the Attribute Names ---
attribute_names = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
    'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry',
    'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
    'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
    'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
    'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
    'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
    'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
]

# --- 4. Main Processing Loop ---
results = []
photo_filenames = os.listdir(CUFS_PHOTOS_DIR)

for filename in tqdm(photo_filenames, desc="Generating Descriptions"):
    photo_path = os.path.join(CUFS_PHOTOS_DIR, filename)

    try:
        image = Image.open(photo_path).convert('RGB')
        image_tensor = data_transform(image).unsqueeze(0).to(DEVICE)

        # Disables gradient calculation for efficiency during inference
        with torch.no_grad():
            outputs = model(image_tensor)
            preds = torch.sigmoid(outputs) > 0.5

        predicted_attributes = [attribute_names[i] for i, pred in enumerate(preds.squeeze()) if pred]
        description_string = ", ".join(predicted_attributes)

        # The CUFS dataset has photos and sketches with matching filenames
        results.append([filename, description_string])

    except Exception as e:
        print(f"Could not process {filename}. Error: {e}")

# --- 5. Save the Results to a CSV File ---
df = pd.DataFrame(results, columns=['sketch_filename', 'description'])
df.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"\nSuccessfully generated descriptions for {len(df)} images.")
print(f"Results saved to {OUTPUT_CSV_PATH}")