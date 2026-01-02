from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import sys
import os
import pandas as pd
import random
import base64
from io import BytesIO
from typing import List, Optional

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src import config
from src.models import Generator
from src.utils import load_vocab, text_to_labels

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
model = None
vocab = None
descriptions_df = None

class GenerateRequest(BaseModel):
    attributes: List[str]

@app.on_event("startup")
async def startup_event():
    global model, vocab, descriptions_df
    print("Loading resources...")
    
    # Load Vocab
    if os.path.exists(config.VOCAB_FILE):
        vocab = load_vocab(config.VOCAB_FILE)
    else:
        print("Warning: vocab.json not found.")
        vocab = {}

    # Load Generator
    if os.path.exists(config.GENERATOR_PATH):
        try:
            model = Generator(config.LATENT_DIM, len(vocab), config.CHANNELS_IMG, config.IMAGE_SIZE).to(config.DEVICE)
            model.load_state_dict(torch.load(config.GENERATOR_PATH, map_location=config.DEVICE))
            model.eval()
            print("Generator loaded.")
        except Exception as e:
            print(f"Error loading generator: {e}")
    else:
        print("Warning: generator.pth not found.")

    # Load Database (Descriptions)
    if os.path.exists(config.DESCRIPTIONS_FILE):
        descriptions_df = pd.read_csv(config.DESCRIPTIONS_FILE)
        # Ensure description is string
        descriptions_df['description'] = descriptions_df['description'].astype(str)
        print(f"Database loaded with {len(descriptions_df)} entries.")
    else:
        print("Warning: descriptions file not found.")

@app.get("/attributes")
def get_attributes():
    if not vocab:
        return []
    return list(vocab.keys())

@app.post("/generate")
def generate_sketch(request: GenerateRequest):
    if not model or not vocab:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Create label vector
    # We can use the existing utils function, but we need to reconstruct the "text description" string
    # or just manually build the vector. Manual is safer/faster.
    
    label_tensor = torch.zeros(len(vocab)).to(config.DEVICE)
    for attr in request.attributes:
        if attr in vocab:
            label_tensor[vocab[attr]] = 1
    
    # Add batch dimension
    label_tensor = label_tensor.unsqueeze(0)
    
    with torch.no_grad():
        noise = torch.randn(1, config.LATENT_DIM).to(config.DEVICE)
        fake_img = model(noise, label_tensor)
        
        # Post-process
        # Normalize from [-1, 1] to [0, 1]
        fake_img = (fake_img + 1) / 2
        fake_img = fake_img.clamp(0, 1)
        fake_img = fake_img.cpu().squeeze(0) # Remove batch
        
        # Convert to PIL
        from torchvision.transforms import ToPILImage
        to_pil = ToPILImage()
        img = to_pil(fake_img)
        
        # Encode to Base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {"image": img_str}

@app.post("/search")
def search_database(request: GenerateRequest):
    if descriptions_df is None:
        raise HTTPException(status_code=503, detail="Database not loaded")
    
    query_attrs = set(request.attributes)
    results = []
    
    for idx, row in descriptions_df.iterrows():
        desc = row['description']
        row_attrs = set([x.strip() for x in desc.split(',')])
        
        # Calculate overlap (Jaccard similarity or simple intersection)
        # Let's do a simple "match score": number of matching attributes
        match_count = len(query_attrs.intersection(row_attrs))
        
        if match_count > 0:
            results.append({
                "filename": row['sketch_filename'],
                "score": match_count,
                "attributes": list(row_attrs)
            })
    
    # Sort by score descending
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Return top 20
    return results[:20]

# Serve Real Photos
# We mount the static directory
app.mount("/photos", StaticFiles(directory=config.MATCHED_PHOTOS_DIR), name="photos")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
