import streamlit as st
import clip
import torch

# Cache to only load model once
@st.cache_resource
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def load_custom_clip_model(model_path="clip_finetuned.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the full saved model
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()  # Ensure it's in inference mode

    _, preprocess = clip.load("ViT-B/32", device=device)

    return model, preprocess, device

# Return vector embeddings for given text (caption)
def get_text_features(captions, device, model):
    text_tokens = clip.tokenize(captions).to(device)
    with torch.no_grad():  
        text_features = model.encode_text(text_tokens)

    return text_features # -> dim: (num_captions, 512)

# Return vector embeddings for given image
def get_image_features(image, device, model, preprocess):
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input) 

    return image_features # -> dim: (1, 512)

# Compute cosine similarity of image & text features
def compute_similarity(image_features, text_features):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    similarity = (image_features @ text_features.T).squeeze(0)
    
    return similarity




