import torch
import json
from torchvision.utils import save_image
import torch.nn as nn
from torchvision.models import resnet34

# --- You must include the Generator class definition ---
# This should be the same class as in your train_gan.py script
class Generator(nn.Module):
    def __init__(self, noise_dim, label_dim, channels_img):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: noise_dim + label_dim -> output size 4x4
            self._block(noise_dim + label_dim, 1024, 4, 1, 0),
            self._block(1024, 512, 4, 2, 1), # 8x8
            self._block(512, 256, 4, 2, 1), # 16x16
            self._block(256, 128, 4, 2, 1), # 32x32
            nn.ConvTranspose2d(128, channels_img, kernel_size=4, stride=2, padding=1), # 64x64
            nn.Tanh() # Output values between -1 and 1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, noise, labels):
        # Concatenate noise and labels
        x = torch.cat([noise, labels], dim=1)
        # Reshape to be used by convolutional layers
        x = x.unsqueeze(2).unsqueeze(3)
        return self.net(x)

# --- CONFIGURATION ---
GENERATOR_PATH = 'generator.pth'
VOCAB_PATH = 'vocab.json'
OUTPUT_FILENAME = 'generated_sketch.png'
LATENT_DIM = 100
CHANNELS_IMG = 1
IMAGE_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- YOUR TEXT DESCRIPTION ---
# Use attribute names from your dataset. Separate them with a comma and a space.
text_description = "Male, Black_Hair, No_Beard, Oval_Face, Wearing_Earrings"

# --- 1. Load Vocabulary and Process Text ---
print(f"Loading vocabulary from {VOCAB_PATH}...")
with open(VOCAB_PATH, 'r') as f:
    vocab = json.load(f)
vocab_size = len(vocab)

# Convert the text description into a multi-hot encoded vector
label_vector = torch.zeros(1, vocab_size, device=DEVICE)
attributes = [attr.strip() for attr in text_description.split(',')]
for attr in attributes:
    if attr in vocab:
        label_vector[0, vocab[attr]] = 1
    else:
        print(f"Warning: Attribute '{attr}' not in vocabulary and will be ignored.")

# --- 2. Load Trained Generator ---
print(f"Loading generator from {GENERATOR_PATH}...")
gen = Generator(LATENT_DIM, vocab_size, CHANNELS_IMG).to(DEVICE)
gen.load_state_dict(torch.load(GENERATOR_PATH, map_location=DEVICE))
gen.eval() # Set to evaluation mode

# --- 3. Generate and Save the Sketch ---
print("Generating sketch...")
with torch.no_grad():
    # Create a random noise vector
    noise = torch.randn(1, LATENT_DIM).to(DEVICE)
    # Generate the image
    generated_image = gen(noise, label_vector)
    
# Save the image
save_image(generated_image, OUTPUT_FILENAME, normalize=True)
print(f"Sketch saved to {OUTPUT_FILENAME}")