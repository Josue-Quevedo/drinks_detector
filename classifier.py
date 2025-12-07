"""
Zero-shot beverage classifier using CLIP.
No training required.

Run:
    
    python classifier.py --image path/to/image.jpg
By: Ulises Josue Quevedo Hernandez 
Date: 12/6/2025
"""

import argparse
from pathlib import Path
import torch
import clip
from PIL import Image

def predict_beverage(image_path: str):
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Labels we want CLIP to classify
    labels = ["cola soda", "orange juice", "bottle of water"]

    # Load image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # Encode text prompts
    text = clip.tokenize(labels).to(device)

    # Compute similarities
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        # Compute cosine similarity
        similarity = (image_features @ text_features.T).softmax(dim=-1)

    # Get best prediction
    best_index = similarity.argmax().item()
    confidence = similarity[0][best_index].item() * 100
    predicted_label = labels[best_index]

    print(f"Prediction: {predicted_label} ({confidence:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot beverage classifier")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    args = parser.parse_args()

    predict_beverage(args.image)
