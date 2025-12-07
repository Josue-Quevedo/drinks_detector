"""
Zero-shot beverage classifier using CLIP.
This script classifies an image into one of three beverages:
cola soda, orange juice, or water — without any training.

Run from terminal:

    python classifier.py --image path/to/image.jpg

Author: Ulises Josue Quevedo Hernandez
Date: 12/6/2025
Edit: 12/7/2025
Version 1.2
"""

import argparse
import torch
import clip
from PIL import Image


def predict_beverage(image_path: str):
    """
    Predicts the beverage category of an image using CLIP in zero-shot mode.

    Args:
        image_path (str): Path to the image to classify.

    Returns:
        None. Prints the predicted label and confidence score.
    """

    # Select device (GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLIP pretrained model and image preprocessing pipeline
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Text labels that represent the beverage categories
    labels = ["cola soda", "orange juice", "bottle of water"]

    # Load and preprocess the image (resize, normalize, tensor conversion)
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # Convert labels into CLIP tokenized text embeddings
    text = clip.tokenize(labels).to(device)

    # Compute CLIP similarities between image and text
    with torch.no_grad():
        # Encode image and text into feature vectors
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        # Compute similarity via dot product + softmax (probability distribution)
        similarity = (image_features @ text_features.T).softmax(dim=-1)

    # Find the index of the highest similarity score
    best_index = similarity.argmax().item()

    # Confidence value (percentage)
    confidence = similarity[0][best_index].item() * 100

    # Final predicted label
    predicted_label = labels[best_index]

    print(f"Prediction: {predicted_label}; Confidence: {confidence:.2f}%")



if __name__ == "__main__":
    # Argument parser for command-line usage
    parser = argparse.ArgumentParser(description="Zero-shot beverage classifier")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    args = parser.parse_args()

    # Run the prediction
    predict_beverage(args.image)
