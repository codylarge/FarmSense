classes = {
    "corn": "Corn is a tall plant with long, narrow green leaves and yellow or golden kernels arranged in rows on cobs. The kernels are surrounded by a husk.",
    "wheat": "Wheat is a grass-like plant with long, slender stems and clusters of small, tan or golden-colored grains at the top.",
    "rice": "Rice is a cereal plant with long, narrow green leaves and small, oval-shaped grains that turn golden as they ripen.",
    "cotton": "Cotton is a plant with broad green leaves and white or creamy cotton fibers that are harvested from its seed bolls.",
    "watermelon": "Watermelon is a large, round or oval fruit with a thick green rind and sweet, red or pink flesh inside, dotted with black seeds.",
    "tomato": "Tomato is a round, red or sometimes green fruit with glossy skin and a green stem. It typically grows to about 3-4 inches in diameter.",
    "carrot": "Carrot is a root vegetable with a long, tapering orange root and feathery green leaves on top.",
    "pineapple": "Pineapple is a tropical fruit with a rough, spiky, brownish-yellow skin and sweet, tangy yellow flesh inside.",
    "peach": "Peach is a round fruit with a fuzzy, yellow-red skin and a sweet, juicy, yellow or orange pulp inside.",
    "sunflower": "Sunflower is a tall, sturdy plant with large yellow petals and a brown center filled with seeds. The leaves are broad and the stalk is thick.",
}

disease_classes = {
    "blight": "Blight is a plant disease that causes rapid wilting, browning, and decay of leaves, stems, and fruit. It may also produce dark lesions.",
    "rust": "Rust is a fungal disease characterized by orange or reddish-brown spots on leaves, stems, and sometimes fruit.",
    "powdery mildew": "Powdery mildew is a fungal infection that creates a white, powdery coating on the surface of leaves, stems, and flowers.",
    "downy mildew": "Downy mildew causes yellow or brown spots on leaves and a fuzzy, grayish growth on the underside of the affected areas.",
    "aphids": "Aphids are tiny, soft-bodied insects that can be green, yellow, or black. They gather on the undersides of leaves and suck out plant sap.",
    "leaf spot": "Leaf spot is a plant disease that causes small, round, or irregular dark spots on the leaves, often surrounded by yellow halos."
}

def get_candidate_captions():
    candidate_captions = [f"{cls}: {desc}" for cls, desc in classes.items()]
    candidate_captions.extend([f"Disease {cls}: {desc}" for cls, desc in disease_classes.items()])
    return candidate_captions
