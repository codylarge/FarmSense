custom_captions = {
    "Healthy Wheat": "Green wheat plant with smooth leaves. No spots, lesions, or yellowing. Leaves are upright, and stems are strong.",
    "Wheat Brown Rust": "Wheat leaves with round, reddish-brown spots and pustules. Some leaves show yellowing and curling.",
    "Wheat Yellow Rust": "Wheat leaves with long yellow streaks and bright yellow pustules. Some leaves curl or dry.",
    "Healthy Rice": "Green rice leaf with no spots, streaks, or damage. Smooth surface, uniform color, and no curling.",
    "Rice Brown Spot": "Rice leaf with brown circular spots, dark edges, and gray centers. Some areas show yellowing or drying.",
    "Rice Leaf Blast": "Rice leaf with long, narrow lesions with dark brown edges and pale centers. Some edges appear torn.",
    "Rice Neck Blast": "Dark lesions on rice panicle neck, leading to weak stems. Grains may be shriveled or discolored.",
    "Potato Early Blight": "Potato leaf with dark brown circular spots with yellow halos. Some spots show concentric rings.",
    "Potato Late Blight": "Potato leaf with dark, irregular lesions. Some areas are water-soaked, with white fungal growth underneath.",
    "Corn Grey Leaf Spot": "Corn leaf with long, gray lesions along veins. Some areas appear dry, with leaf curling in severe cases.",
    "Corn Healthy": "Green corn leaf with a smooth surface, intact edges, and no lesions or discoloration.",
    "Corn Northern Leaf Blight": "Corn leaf with long, grayish-brown elliptical lesions. Some areas appear dry or brittle.",
    "Corn Common Rust": "Corn leaf with raised, reddish-brown pustules scattered across the surface, often with yellow halos."
}

def get_classes(crop="all"):
    if crop == "Corn":
        return {k: v for k, v in custom_captions.items() if "Corn" in k}
    elif crop == "Wheat":
        return {k: v for k, v in custom_captions.items() if "Wheat" in k}
    elif crop == "Rice":
        return {k: v for k, v in custom_captions.items() if "Rice" in k}
    elif crop == "Potato":
        return {k: v for k, v in custom_captions.items() if "Potato" in k}
    else:
        return custom_captions
'''
# Function to return classes based on crop selection
def get_candidate_captions(crop="Select a Crop"):
    crop_classes = {
        "Wheat": wheat_classes,
        "Rice": rice_classes,
        "Corn": corn_classes,
        "Potato": potato_classes,
        "Select a Crop": all_classes  # This contains all classes
    }
    
    return [f"{cls}: {desc}" for cls, desc in crop_classes.get(crop, {}).items()]  # Ensure .items() is used
'''