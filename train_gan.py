import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm
import os
import sys

# Add project root to path so we can import src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.models import Generator, Discriminator, weights_init_normal
from src.dataset import get_dataloader
from src.utils import save_vocab

def train():
    print(f"--- Starting Training on {config.DEVICE} ---")
    
    # Ensure output directory exists
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 1. Load Data
    dataloader, dataset = get_dataloader(
        config.DESCRIPTIONS_FILE, 
        config.MATCHED_SKETCHES_DIR, 
        config.BATCH_SIZE
    )
    print(f"Vocabulary Size: {dataset.vocab_size}")
    
    # Save vocab immediately for inference
    save_vocab(dataset.vocab, config.VOCAB_FILE)

    # 2. Initialize Models
    gen = Generator(config.LATENT_DIM, dataset.vocab_size, config.CHANNELS_IMG, config.IMAGE_SIZE).to(config.DEVICE)
    disc = Discriminator(config.CHANNELS_IMG, dataset.vocab_size, config.IMAGE_SIZE).to(config.DEVICE)

    # Apply custom weights initialization
    gen.apply(weights_init_normal)
    disc.apply(weights_init_normal)

    # Load checkpoint if exists
    start_epoch = 0
    if os.path.exists(config.GENERATOR_PATH):
        print(f"Loading checkpoint from {config.GENERATOR_PATH}...")
        try:
            gen.load_state_dict(torch.load(config.GENERATOR_PATH, map_location=config.DEVICE))
            if os.path.exists(config.DISCRIMINATOR_PATH):
                disc.load_state_dict(torch.load(config.DISCRIMINATOR_PATH, map_location=config.DEVICE))
                print("Discriminator weights loaded.")
            else:
                print("No discriminator checkpoint found. Using random init for D.")
            
            print("Resuming training with loaded weights.")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")

    # 3. Optimizers (TTUR: Two-Time-Scale Update Rule)
    # Generator gets higher LR to learn faster than Discriminator
    opt_gen = optim.Adam(gen.parameters(), lr=0.0004, betas=(config.BETA1, config.BETA2))
    opt_disc = optim.Adam(disc.parameters(), lr=0.0001, betas=(config.BETA1, config.BETA2))

    # 4. Loss Function
    criterion = nn.BCELoss()
    criterion_l1 = nn.L1Loss()

    # Fixed noise for visualization
    fixed_noise = torch.randn(16, config.LATENT_DIM).to(config.DEVICE)
    # Get a fixed set of labels for consistent visualization
    # We'll just take the first batch's labels or random ones if batch is small
    fixed_labels = next(iter(dataloader))[1][:16].to(config.DEVICE)
    if fixed_labels.size(0) < 16:
        # If batch size is small, pad or just use what we have
        pass 

    # 5. Training Loop
    for epoch in range(config.NUM_EPOCHS):
        gen.train()
        disc.train()
        
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        for batch_idx, (real_imgs, labels) in enumerate(loop):
            real_imgs = real_imgs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            batch_size = real_imgs.size(0)

            # --- Train Discriminator ---
            opt_disc.zero_grad()
            
            # Real Images
            # Soft labels: 0.9 for real instead of 1.0
            real_targets = torch.full((batch_size, 1), config.LABEL_SMOOTHING, device=config.DEVICE)
            disc_real, _ = disc(real_imgs, labels) # Ignore features for D training
            loss_disc_real = criterion(disc_real, real_targets)
            
            # Fake Images
            noise = torch.randn(batch_size, config.LATENT_DIM).to(config.DEVICE)
            fake_imgs = gen(noise, labels)
            fake_targets = torch.zeros((batch_size, 1), device=config.DEVICE)
            disc_fake, _ = disc(fake_imgs.detach(), labels) # Detach to avoid G gradients
            loss_disc_fake = criterion(disc_fake, fake_targets)
            
            # Total D Loss
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            loss_disc.backward()
            opt_disc.step()

            # --- Train Generator ---
            opt_gen.zero_grad()
            
            # Generator wants Discriminator to think images are real (1.0)
            output, fake_features = disc(fake_imgs, labels)
            loss_adv = criterion(output, torch.ones_like(output))
            
            # Feature Matching Loss
            # We want the fake image's features to match the real image's features
            # This forces the Generator to produce structurally "real" characteristics
            _, real_features = disc(real_imgs, labels)
            
            loss_fm = 0
            for fake_feat, real_feat in zip(fake_features, real_features):
                loss_fm += criterion_l1(fake_feat, real_feat.detach())
            
            # Scale Feature Matching loss (typically 10.0 or determined empirically)
            loss_gen = loss_adv + (10 * loss_fm)
            
            loss_gen.backward()
            opt_gen.step()
            
            # Update progress bar
            loop.set_postfix(loss_d=loss_disc.item(), loss_g=loss_adv.item(), fm=loss_fm.item())

        # --- End of Epoch ---
        # 1. Save Sample Images
        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                gen.eval()
                fake_samples = gen(fixed_noise, fixed_labels)
                # Invert for visualization (Black ink on White BG)
                fake_samples = -fake_samples
                save_image(fake_samples, f"{config.OUTPUT_DIR}/epoch_{epoch+1}.png", normalize=True)
                gen.train()

        # 2. Save Model Checkpoint (Every 50 epochs or last)
        if (epoch + 1) % 50 == 0 or (epoch + 1) == config.NUM_EPOCHS:
            torch.save(gen.state_dict(), config.GENERATOR_PATH)
            torch.save(disc.state_dict(), config.DISCRIMINATOR_PATH)
            print(f"Saved models to {config.GENERATOR_PATH} and {config.DISCRIMINATOR_PATH}")

if __name__ == "__main__":
    train()
