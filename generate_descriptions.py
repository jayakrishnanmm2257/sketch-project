import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import sys
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.models import get_attribute_classifier

def main():
    print(f"--- Generating Descriptions on {config.DEVICE} ---")
    
    # 1. Load Model
    if not os.path.exists(config.CLASSIFIER_PATH):
        print(f"Error: Classifier weights not found at {config.CLASSIFIER_PATH}")
        return

    model = get_attribute_classifier(num_attributes=40, weights_path=config.CLASSIFIER_PATH, device=config.DEVICE)
    model.to(config.DEVICE)
    model.eval()

    # 2. Define Image Transformations (Must match training of classifier)
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Attributes List (Standard CelebA)
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

    # Attributes to exclude because they are color-dependent or unreliable in grayscale/sketches
    EXCLUDED_ATTRIBUTES = {
        'Pale_Skin', 'Rosy_Cheeks', 'Blurry', 'Wearing_Lipstick', 'Heavy_Makeup', 'Attractive',
        'Wearing_Necktie', 'Wearing_Necklace', 'Wearing_Hat'
    }

    # 4. Process Images
    results = []
    if not os.path.exists(config.MATCHED_PHOTOS_DIR):
        print(f"Error: Photos directory not found at {config.MATCHED_PHOTOS_DIR}")
        return
        
    photo_filenames = os.listdir(config.MATCHED_PHOTOS_DIR)

    for filename in tqdm(photo_filenames, desc="Processing"):
        photo_path = os.path.join(config.MATCHED_PHOTOS_DIR, filename)

        try:
            image = Image.open(photo_path).convert('RGB')
            image_tensor = data_transform(image).unsqueeze(0).to(config.DEVICE)

            with torch.no_grad():
                outputs = model(image_tensor)
                # Lower threshold to capture more subtle attributes
                preds = torch.sigmoid(outputs) > 0.3

            predicted_attributes = [
                attribute_names[i] for i, pred in enumerate(preds.squeeze()) 
                if pred and attribute_names[i] not in EXCLUDED_ATTRIBUTES
            ]
            
            # --- Explicit Defaults (Fill in the blanks) ---
            # 1. Gender (Existing logic)
            if 'Male' not in predicted_attributes:
                predicted_attributes.append('Female')
            
            # 2. Expression
            if 'Smiling' not in predicted_attributes and 'Mouth_Slightly_Open' not in predicted_attributes:
                predicted_attributes.append('Neutral_Expression')

            # 3. Nose
            if 'Big_Nose' not in predicted_attributes and 'Pointy_Nose' not in predicted_attributes:
                predicted_attributes.append('Average_Nose')

            # 4. Eyewear
            if 'Eyeglasses' not in predicted_attributes:
                predicted_attributes.append('No_Eyeglasses')

            # 5. Lips
            if 'Big_Lips' not in predicted_attributes:
                predicted_attributes.append('Average_Lips')

            # 6. Eyebrows
            if 'Arched_Eyebrows' not in predicted_attributes and 'Bushy_Eyebrows' not in predicted_attributes:
                predicted_attributes.append('Average_Eyebrows')

            description_string = ", ".join(predicted_attributes)

            results.append([filename, description_string])

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # 5. Save Results
    df = pd.DataFrame(results, columns=['sketch_filename', 'description'])
    df.to_csv(config.DESCRIPTIONS_FILE, index=False)
    print(f"Saved descriptions to {config.DESCRIPTIONS_FILE}")

if __name__ == "__main__":
    main()
