import os
import torch
import pandas as pd
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .config import IMAGE_SIZE, CHANNELS_IMG, BATCH_SIZE

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
            if isinstance(desc, str):
                attrs = [attr.strip() for attr in desc.split(',')]
                all_attrs.update(attrs)
        # Sort for deterministic behavior
        sorted_attrs = sorted(list(all_attrs))
        return {attr: i for i, attr in enumerate(sorted_attrs)}

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        filename = self.labels_df.iloc[idx, 0]
        # Handle potential differences in extension or path
        sketch_path = os.path.join(self.sketch_dir, filename)
        
        # Fallback for extension mismatch if necessary (though our prep script handles it)
        if not os.path.exists(sketch_path):
             sketch_path = os.path.join(self.sketch_dir, filename.replace('.jpg', '.png'))

        image = Image.open(sketch_path).convert("L")  # Convert to grayscale
        
        # Invert image: Black background (0), White ink (255)
        # This creates a sparse representation which is easier for GANs to learn
        image = ImageOps.invert(image)
        
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

def get_dataloader(csv_file, sketch_dir, batch_size=BATCH_SIZE, num_workers=0):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # Ensure tuple
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) # Range [-1, 1]
    ])

    dataset = SketchDataset(csv_file=csv_file, sketch_dir=sketch_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return dataloader, dataset
