wheat_classes = {
    "Healthy Wheat": "Green wheat plant with smooth leaves. No spots, lesions, or yellowing.",
    "Wheat Brown Rust": "Wheat leaves with round, reddish-brown spots and pustules.",
    "Wheat Yellow Rust": "Wheat leaves with long yellow streaks and bright yellow pustules.",
}

rice_classes = {
    "Healthy Rice": "Green rice leaf with no spots, streaks, or damage.",
    "Rice Brown Spot": "Rice leaf with brown circular spots, dark edges, and gray centers.",
    "Rice Leaf Blast": "Rice leaf with long, narrow lesions with dark brown edges and pale centers.",
    "Rice Neck Blast": "Dark lesions on rice panicle neck, leading to weak stems.",
}

potato_classes = {
    "Potato Early Blight": "Potato leaf with dark brown circular spots with yellow halos.",
    "Potato Late Blight": "Potato leaf with dark, irregular lesions.",
}

corn_classes = {
    "Corn Grey Leaf Spot": "Corn leaf with long, gray lesions along veins.",
    "Corn Healthy": "Green corn leaf with a smooth surface, intact edges.",
    "Corn Northern Leaf Blight": "Corn leaf with long, grayish-brown elliptical lesions.",
    "Corn Common Rust": "Corn leaf with raised, reddish-brown pustules.",
}

# 🟢 Combine all classes into a single dictionary (if needed)
all_classes = {**wheat_classes, **rice_classes, **potato_classes, **corn_classes}


# Function to return classes based on crop selection
def get_candidate_captions(crop="Select a Crop"):
    crop_classes = {
        "Wheat": wheat_classes,
        "Rice": rice_classes,
        "Corn": corn_classes,
        "Potato": potato_classes,
        "Select a Crop": all_classes  # This contains all classes
    }
    
    return [f"{cls}: {desc}" for cls, desc in crop_classes.get(crop, {}).items()]  # ✅ Ensure .items() is used
