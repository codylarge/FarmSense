classes = {
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
'''
disease_classes = {
    "powdery mildew": "Powdery mildew is a fungal infection that creates a white, powdery coating on the surface of leaves, stems, and flowers.",
}
'''

def get_candidate_captions():
    candidate_captions = [f"{cls}: {desc}" for cls, desc in classes.items()]
    #candidate_captions.extend([f"Disease {cls}: {desc}" for cls, desc in disease_classes.items()])
    return candidate_captions
