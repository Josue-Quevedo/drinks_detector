"""
Zero-shot beverage classifier using CLIP.
This script classifies three images (soda, orange juice, water)
and calculates the individual accuracy and the average accuracy.

Run from terminal:

    python classifier.py

Author: Ulises Josue Quevedo Hernandez
Date: 12/6/2025
Last edit: 12/8/2025
Version 1.3
"""

import torch
import clip
from PIL import Image
from typing import List


def predict_beverage(image_path: str, model, preprocess, device: str) -> float:
    """
    Predicts the beverage category of an image using CLIP in zero-shot mode.

    Args:
        image_path (str): Path to the image to classify.
        model: Loaded CLIP model.
        preprocess: Preprocessing pipeline.
        device (str): CPU or GPU device.

    Returns:
        float: Confidence percentage of the predicted label.
    """

    # Text labels that represent the beverage categories
    labels = ["cola soda", "orange juice", "bottle of water"]

    # Load and preprocess the image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # Convert labels into CLIP tokenized text embeddings
    text = clip.tokenize(labels).to(device)

    # Compute CLIP similarities
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        similarity = (image_features @ text_features.T).softmax(dim=-1)

    # Best prediction index
    best_index = similarity.argmax().item()

    # Confidence (percentage)
    confidence = similarity[0][best_index].item() * 100

    predicted_label = labels[best_index]

    print(f"[{image_path}] -> Prediction: {predicted_label}; Confidence: {confidence:.2f}%")

    return confidence


if __name__ == "__main__":
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLIP
    model, preprocess = clip.load("ViT-B/32", device=device)

    # List of the 3 images to classify
    image_list: List[str] = [
        "examples/soda1.jpg",
        "examples/juice1.jpg",
        "examples/water1.jpg"
    ]

    print("\n=== Evaluating 3 Beverage Images ===\n")

    # Store accuracies
    accuracies = []

    for img_path in image_list:
        confidence = predict_beverage(img_path, model, preprocess, device)
        accuracies.append(confidence)

    # Compute and print average accuracy
    average_accuracy = sum(accuracies) / len(accuracies)

    print("\n=== Results Summary ===")
    for i, acc in enumerate(accuracies, 1):
        print(f"Image {i} Accuracy: {acc:.2f}%")

    print(f"\nFinal Average Accuracy: {average_accuracy:.2f}%")