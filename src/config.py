import os
import torch

# --- PROJECT ROOT ---
# Assuming this config is in src/config.py, the root is one level up
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- DATA PATHS ---
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MATCHED_PHOTOS_DIR = os.path.join(DATA_DIR, 'matched_photos_original')
MATCHED_SKETCHES_DIR = os.path.join(DATA_DIR, 'matched_sketches_original')
DESCRIPTIONS_FILE = os.path.join(PROJECT_ROOT, 'sketch_descriptions_original.csv')
VOCAB_FILE = os.path.join(PROJECT_ROOT, 'vocab.json')

# --- OUTPUT PATHS ---
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'gan_outputs')
GENERATOR_PATH = os.path.join(PROJECT_ROOT, 'generator.pth')
DISCRIMINATOR_PATH = os.path.join(PROJECT_ROOT, 'discriminator.pth')
GENERATED_IMG_PATH = os.path.join(PROJECT_ROOT, 'generated_sketch.png')

# --- HYPERPARAMETERS ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256
CHANNELS_IMG = 1
LATENT_DIM = 100
BATCH_SIZE = 16
NUM_EPOCHS = 200
LEARNING_RATE = 0.0002 # Increased slightly as commonly used for DCGAN-like
BETA1 = 0.5
BETA2 = 0.999
LABEL_SMOOTHING = 0.9

# --- ATTRIBUTE CLASSIFIER ---
CLASSIFIER_PATH = os.path.join(PROJECT_ROOT, 'facial_attribute_classifier.pth')