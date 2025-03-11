import streamlit as st
import clip
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
from src.classes import get_classes

# __all__ = ["load_custom_clip_model", "get_text_features", "get_image_features", "compute_similarity", "classify_image"]


import torch.nn.functional as F

class CLIPFineTuner(nn.Module):
    def __init__(self, model, num_classes):
        super(CLIPFineTuner, self).__init__()
        self.model = model
        self.classifier = nn.Linear(model.visual.output_dim, num_classes)  # Fine-tuned classifier

    def forward(self, x):
        features = self.model.encode_image(x).float()  # Extract image features
        features = F.normalize(features, dim=-1)  # Normalize features
        return self.classifier(features)  # Predict class logits

    def encode_text(self, text):
        return self.model.encode_text(text)
    
    def encode_image(self, image):
        return self.model.encode_image(image)

def load_basic_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def load_custom_clip_model(model_path="clip_finetuned.pth", num_classes=13):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load original CLIP model
    base_model, _ = clip.load("ViT-B/32", device=device)

    # Wrap it in CLIPFineTuner
    model = CLIPFineTuner(base_model, num_classes).to(device)

    # Load the saved state_dict (instead of the full model)
    state_dict = torch.load(model_path, map_location=device)

    if 'state_dict' in state_dict:  # If saved inside a dict
        state_dict = state_dict['state_dict']

    model.load_state_dict(state_dict)  # Load weights
    model.to(device)
    model.eval()

    # Define a preprocessing pipeline (must match training-time transforms)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])
    ])

    #print("Custom mode:", model)
    return model, preprocess, device

def get_text_features(captions, device, model):
    text_tokens = clip.tokenize(captions).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    return text_features

def get_image_features(image, device, model, preprocess):
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    return image_features

def compute_similarity(image_features, text_features):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    similarity = (image_features @ text_features.T).squeeze(0)
    
    return similarity

def classify_image(image, model, preprocess, device, custom_captions):
    # Get candidate captions and their features
    text_inputs = torch.cat([clip.tokenize(custom_captions[c]) for c in custom_captions.keys()]).to(device)
    
    image = preprocess(image).unsqueeze(0).to(device)  # Apply CLIP preprocessing

    # Extract image and text features
    image_features = model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)

    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute similarity between image and text features
    similarity = image_features @ text_features.T  # Shape: (1, num_classes)

    # Find the caption with the highest similarity score
    predicted_index = similarity.argmax(dim=-1).item()  # Convert tensor to int
    predicted_label = list(custom_captions.keys())[predicted_index]
    best_similarity_score = similarity[0, predicted_index].item()  # Extract top similarity score

    return predicted_label, best_similarity_score  # String, Number
