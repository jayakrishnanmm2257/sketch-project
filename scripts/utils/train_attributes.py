import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet34, ResNet34_Weights
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm

# --- Configuration ---
# IMPORTANT: Update these paths to where you saved your CelebA data
# Use raw strings (r'...') to avoid errors with backslashes on Windows
IMG_DIR = 'C:/Users/krish/Downloads/Sketch Project/data/CelebA/img_align_celeba'
ATTR_PATH = 'C:/Users/krish/Downloads/Sketch Project/data/CelebA/list_attr_celeba.txt'
PARTITION_PATH = 'C:/Users/krish/Downloads/Sketch Project/data/CelebA/list_eval_partition.txt'

# Model and training parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20 # Start with a few epochs to test the pipeline

# --- 1. Custom Dataset Definition ---
class CelebADataset(Dataset):
    def __init__(self, img_dir, attr_path, partition_path, split, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # Read attribute and partition files using the corrected separator
        attr_df = pd.read_csv(attr_path, sep=r'\s+', header=1)
        partition_df = pd.read_csv(partition_path, sep=r'\s+', header=None, names=['image_id', 'partition'])
        
        # Define which partition to use (0=train, 1=validation, 2=test)
        split_map = {'train': 0, 'val': 1, 'test': 2}
        target_partition = split_map[split]
        
        # Filter image IDs for the chosen split
        image_ids = partition_df[partition_df['partition'] == target_partition]['image_id'].tolist()
        
        # Filter attributes for the chosen split and convert to tensors
        self.attributes = attr_df[attr_df.index.isin(image_ids)]
        self.image_list = self.attributes.index.tolist()
        
        # Convert labels from -1/1 to 0/1 and store as a tensor
        self.labels = torch.tensor(self.attributes.values == 1, dtype=torch.float32)
        self.attr_names = list(self.attributes.columns)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.image_list[idx])
        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# --- 3. Model Definition ---
def get_model(num_attributes):
    # Use a pre-trained ResNet34 model
    model = resnet34(weights=ResNet34_Weights.DEFAULT)
    
    # Freeze all layers in the network
    for param in model.parameters():
        param.requires_grad = False
        
    # Get the number of input features for the final layer
    num_ftrs = model.fc.in_features
    
    # Replace the final layer for our multi-label classification task
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_attributes)
    )
    return model

# --- 4. Accuracy Calculation Function ---
def calculate_accuracy(outputs, labels):
    # Apply sigmoid and threshold to get binary predictions
    preds = torch.sigmoid(outputs) > 0.5
    correct = (preds == labels).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy


# --- Main execution block ---
# This 'if' statement is crucial for multiprocessing on Windows
if __name__ == '__main__':
    print(f"Using device: {DEVICE}")

    # --- Data Transformations and Loaders ---
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CelebADataset(IMG_DIR, ATTR_PATH, PARTITION_PATH, 'train', data_transform)
    val_dataset = CelebADataset(IMG_DIR, ATTR_PATH, PARTITION_PATH, 'val', data_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")
    print(f"Number of attributes: {len(train_dataset.attr_names)}")

    model = get_model(num_attributes=len(train_dataset.attr_names)).to(DEVICE)

    # --- Loss Function and Optimizer ---
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    # --- Training and Validation Loop ---
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for images, labels in train_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_acc += calculate_accuracy(outputs, labels).item()
            train_bar.set_postfix(loss=running_loss/len(train_bar), acc=running_acc/len(train_bar))

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_acc += calculate_accuracy(outputs, labels).item()
                val_bar.set_postfix(loss=val_loss/len(val_bar), acc=val_acc/len(val_bar))
                
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> "
              f"Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {running_acc/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc/len(val_loader):.4f}")

    # --- Save the Model ---
    torch.save(model.state_dict(), 'facial_attribute_classifier.pth')
    print("Finished Training. Model saved to facial_attribute_classifier.pth")