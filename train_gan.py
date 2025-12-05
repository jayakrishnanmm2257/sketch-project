import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
import numpy as np

# --- 1. CONFIGURATION ---
DATASET_CSV = 'sketch_descriptions.csv'
SKETCH_DIR = r'C:\Users\krish\Downloads\Sketch Project\data\CUHK\sketches' # IMPORTANT: Update this path
OUTPUT_DIR = 'gan_outputs' # Folder to save generated images
NUM_EPOCHS = 2000 # GANs need many epochs. Start with 300-500 for a test.
BATCH_SIZE = 16 # Keep this low due to small dataset and model size
LEARNING_RATE = 0.0002
BETA1 = 0.5 # A standard hyperparameter for the Adam optimizer in GANs
LATENT_DIM = 100 # Size of the random noise vector
IMAGE_SIZE = 64 # Size of images to be generated (must be a power of 2)
CHANNELS_IMG = 1 # Grayscale images

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- 2. DATASET AND TEXT PROCESSING ---
class SketchDataset(Dataset):
    def __init__(self, csv_file, sketch_dir, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.sketch_dir = sketch_dir
        self.transform = transform
        self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)

    def _build_vocab(self):
        # Creates a vocabulary of all unique attributes
        all_attrs = set()
        for desc in self.labels_df['description']:
            # Check if it's a valid string before splitting
            if isinstance(desc, str):
                attrs = [attr.strip() for attr in desc.split(',')]
                all_attrs.update(attrs)
        # Create a mapping from attribute to integer index
        return {attr: i for i, attr in enumerate(all_attrs)}

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        sketch_path = os.path.join(self.sketch_dir, self.labels_df.iloc[idx, 0])
        image = Image.open(sketch_path).convert("L")  # Convert to grayscale
        
        description = self.labels_df.iloc[idx, 1]
        
        # Convert text description to a multi-hot encoded vector
        label_vector = torch.zeros(self.vocab_size)
        if isinstance(description, str):
            attrs = [attr.strip() for attr in description.split(',')]
            for attr in attrs:
                if attr in self.vocab:
                    label_vector[self.vocab[attr]] = 1
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_vector

# --- 3. MODELS (GENERATOR & DISCRIMINATOR) ---

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


class Discriminator(nn.Module):
    def __init__(self, channels_img, label_dim):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Linear(label_dim, IMAGE_SIZE*IMAGE_SIZE)
        
        self.net = nn.Sequential(
            # Input: channels_img + 1 (for label channel) -> 32x32
            self._block(channels_img + 1, 128, 4, 2, 1),
            self._block(128, 256, 4, 2, 1), # 16x16
            self._block(256, 512, 4, 2, 1), # 8x8
            self._block(512, 1024, 4, 2, 1), # 4x4
            nn.Conv2d(1024, 1, kernel_size=4, stride=2, padding=0), # 1x1
            nn.Sigmoid() # Output a probability
        )
        
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x, labels):
        # Reshape labels and concatenate them to the image as an extra channel
        label_embedding = self.label_embedding(labels).view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
        x = torch.cat([x, label_embedding], dim=1)
        return self.net(x)

# --- 4. TRAINING SETUP ---
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)])
])

dataset = SketchDataset(csv_file=DATASET_CSV, sketch_dir=SKETCH_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize models
gen = Generator(LATENT_DIM, dataset.vocab_size, CHANNELS_IMG).to(DEVICE)
disc = Discriminator(CHANNELS_IMG, dataset.vocab_size).to(DEVICE)

# Optimizers
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

# Loss function
criterion = nn.BCELoss()

# Fixed noise and labels for visualizing generator's progress
fixed_noise = torch.randn(BATCH_SIZE, LATENT_DIM).to(DEVICE)
# For visualization, let's get a fixed batch of real labels
fixed_labels = next(iter(dataloader))[1].to(DEVICE)


# --- 5. TRAINING LOOP ---
for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
        real = real.to(DEVICE)
        labels = labels.to(DEVICE)
        
        noise = torch.randn(real.size(0), LATENT_DIM).to(DEVICE)
        fake = gen(noise, labels)

        # -- Train Discriminator --
        disc.zero_grad()
        # Train with real images
        disc_real = disc(real, labels).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        # Train with fake images
        disc_fake = disc(fake.detach(), labels).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        # Total discriminator loss
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        loss_disc.backward()
        opt_disc.step()

        # -- Train Generator --
        gen.zero_grad()
        output = disc(fake, labels).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output)) # We want the discriminator to think the fake images are real
        loss_gen.backward()
        opt_gen.step()
        
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}")

    # Save some generated images for inspection
    with torch.no_grad():
        fake_samples = gen(fixed_noise, fixed_labels).detach().cpu()
        save_image(fake_samples, f"{OUTPUT_DIR}/sample_epoch_{epoch+1}.png", normalize=True)

print("Training finished.")

# --- ADD THIS TO THE END OF train_gan.py ---

import json

# Save the trained Generator's state
torch.save(gen.state_dict(), 'generator.pth')
print("Generator model saved to generator.pth")

# Save the vocabulary mapping
with open('vocab.json', 'w') as f:
    json.dump(dataset.vocab, f)
print("Vocabulary saved to vocab.json")