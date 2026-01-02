# Gemini Project Analysis: AI-Powered Sketch Generation

## Project Overview

This project is a deep learning pipeline that generates facial sketches from textual descriptions of attributes. It leverages a two-stage process:

1.  **Facial Attribute Classification:** A pre-trained ResNet-34 model (`facial_attribute_classifier.pth`) is used to classify 40 different facial attributes from a photo.
2.  **Conditional Sketch Generation:** A Conditional Generative Adversarial Network (cGAN) is trained on a dataset of face photos and their corresponding sketches. It learns to generate a sketch from a conditioning vector that represents the facial attributes.

The core technologies are **Python** and **PyTorch**.

---

## Recent Architectural & Training Improvements (Dec 2025)

### 1. "Explicit Defaults" Strategy
To solve the issue of "undefined" or "mishmashed" features (e.g., a face with no nose defined), we implemented a logic to explicitly tag missing attributes:
*   **Expression:** `Neutral_Expression` (if not Smiling/Open Mouth).
*   **Structure:** `Average_Nose`, `Average_Lips`, `Average_Eyebrows` (if no distinctive traits present).
*   **Eyewear:** `No_Eyeglasses` (explicitly stating absence).
*   **Result:** This provides the Generator with a complete "blueprint" for every face, significantly improving structural coherence.

### 2. Feature Matching Loss
*   **Problem:** Pixel-wise loss (L1) often leads to blurry results, while standard GAN loss can be unstable.
*   **Solution:** Refactored the **Discriminator** to return intermediate feature maps. The Generator is now trained to minimize the L1 distance between the *features* of the generated sketch and the real sketch.
*   **Benefit:** This forces the model to learn high-level structures (shapes, edges) rather than just raw pixel averages, resulting in sharper definition.

### 3. Two-Time-Scale Update Rule (TTUR)
*   **Adjustment:** set Generator LR to `0.0004` and Discriminator LR to `0.0001`.
*   **Reason:** Prevents the Discriminator from learning too fast and overpowering the Generator, ensuring stable training over long durations.

### 4. Color Attributes in Grayscale
*   **Insight:** While the output is black-and-white, attributes like `Blond_Hair` vs `Black_Hair` are retained.
*   **Mechanism:** The model successfully learns to map these text tags to **shading density**.
    *   `Blond_Hair` $ightarrow$ Sparse strokes / White space.
    *   `Black_Hair` $ightarrow$ Dense, dark ink strokes.
    *   `Gray_Hair` $ightarrow$ Textural mapping (lines indicating strands).

---

## Dataset Analysis & Preparation

The project requires two main datasets: **CelebA** for attribute training (pre-trained) and a combination of **GrayFERET** (grayscale photos) and **CUHK** (sketches) for the sketch generation task.

### Key Optimization: "Resize & Center Crop"
To solve aspect ratio distortion ("squashed faces"):
*   **Photos:** Center Cropped to 256x256.
*   **Sketches:** Cropped to ink $ightarrow$ Resized width to 256 $ightarrow$ Center Cropped to 256x256.

---

## Building and Running the GAN Pipeline

### Activate Virtual Environment
`.\venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/macOS)

### Step 1: Prepare the Dataset
```bash
python prepare_gan_dataset.py 
```

### Step 2: Generate Attribute Descriptions (Enriched)
Generates descriptions with explicit defaults:
```bash
python generate_descriptions.py
```

### Step 3: Train the Conditional GAN
Train with Feature Matching and TTUR (Recommended: 200 epochs):
```bash
python train_gan.py
```

### Usage
Generate a sketch from a description string:
```bash
python generate_sketch.py
```
*Example:* `generate('Female, Smiling, Eyeglasses, Black_Hair, Young, Average_Nose')`

---

## Current Project State

*   **Model:** Trained for **200 Epochs** with Feature Matching.
*   **Status:** Capable of generating high-definition, structurally coherent sketches.
*   **Next Steps:** Integration into a Web UI (Backend ready in `web_app/backend`, Frontend in `web_app/frontend`).
