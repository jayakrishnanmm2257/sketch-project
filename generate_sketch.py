import torch
from torchvision.utils import save_image
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.models import Generator
from src.utils import load_vocab, text_to_labels

def generate(text_description):
    print(f"--- Generating Sketch ---")
    print(f"Description: {text_description}")
    
    # 1. Load Resources
    if not os.path.exists(config.VOCAB_FILE):
        print("Error: vocab.json not found. Train the model first.")
        return
    if not os.path.exists(config.GENERATOR_PATH):
        print("Error: generator.pth not found. Train the model first.")
        return

    vocab = load_vocab(config.VOCAB_FILE)
    
    # 2. Prepare Model
    gen = Generator(config.LATENT_DIM, len(vocab), config.CHANNELS_IMG, config.IMAGE_SIZE).to(config.DEVICE)
    try:
        gen.load_state_dict(torch.load(config.GENERATOR_PATH, map_location=config.DEVICE))
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        print("The architecture in src/models.py might not match the saved weights.")
        return

    gen.eval()

    # 3. Process Input
    label_vector = text_to_labels(text_description, vocab, config.DEVICE)
    
    # 4. Generate
    with torch.no_grad():
        noise = torch.randn(1, config.LATENT_DIM).to(config.DEVICE)
        fake_img = gen(noise, label_vector)
        
        # Invert colors back to standard (Black ink on White background)
        # Model output: Ink=High(1), BG=Low(-1)
        # We want: Ink=Low(-1), BG=High(1)
        fake_img = -fake_img
        
        save_image(fake_img, config.GENERATED_IMG_PATH, normalize=True)
        print(f"Success! Saved to {config.GENERATED_IMG_PATH}")

if __name__ == "__main__":
    # Example usage
    desc = "Male, Bald, Smiling, Black_Hair, Oval_Face"
    generate(desc)
