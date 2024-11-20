import torch
import clip
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import glob
import os


def download_dataset():
    dataset_path = './plants-classification'

    if not os.path.exists(dataset_path):
        #os.environ['KAGGLE_USERNAME'] = 'yon***oon'
        #os.environ['KAGGLE_KEY'] = '929a975ada6ec3c**5f22***'


        # Download and unzip dataset using Kaggle
        #os.system('pip install kaggle')
        #os.system('kaggle datasets download -d marquis03/plants-classification --unzip -p ./plants-classification')

        print("Dataset downloaded.")
    else:
        print("Dataset already exists. Skipping download.")

    return dataset_path


#  load class names and generate candidate captions as prompt,
def get_class_names(dataset_path):

    class_names = ["guava", "ginger", "soybeans"]
   # candidate_captions = [f"{cls}" for cls in class_names]
    prompt = [f"A picture of {cls}" for cls in class_names]
    print(f"Selected classes: {class_names}")
    print(f"Candidate captions: {prompt}")

    return class_names, prompt

#implemnt pytorch dataset.
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image), self.image_paths[idx]


# function to load images and creat the  dataloader // *remove  the prints
def load_images(dataset_path, class_names):
    test_images = []


    for cls in class_names:
        test_images.extend(glob.glob(os.path.join(dataset_path, 'test', cls, '*.jpg')))

    print(f"Total test images: {len(test_images)}")


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])


    dataset = ImageDataset(test_images, transform)
    return DataLoader(dataset, batch_size=8, shuffle=False)

#  load CLIP model
def load_clip_model(device):
    model, _ = clip.load("ViT-B/32", device=device)
    return model



def evaluate_model(dataloader, model, class_names, text_features, device):
    correct = 0
    total = 0

    for images, image_paths in dataloader:
        images = images.to(device)

        with torch.no_grad():
            image_features = model.encode_image(images)
            logits_per_image = image_features @ text_features.T
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # loop through each image and compare prediction with true class
        for i, img_path in enumerate(image_paths):
            pred = class_names[probs[i].argmax()]
            true_class = os.path.basename(os.path.dirname(img_path))

            # count correct predictions
            correct += (pred == true_class)
            total += 1

    accuracy = correct / total if total > 0 else 0
    print(f'Overall accuracy: {accuracy:.2f}')

# Main function to run the script
def main():
    #get data set
    dataset_path = download_dataset()

    # load class
    class_names, candidate_captions = get_class_names(dataset_path)


    dataloader = load_images(dataset_path, class_names)

    device = "cuda" if torch.cuda.is_available() else "cpu"  #
    model = load_clip_model(device)

    # encode text prompts
    text_tokens = clip.tokenize(candidate_captions).to(device)  # tokenize the prompt and move to the device
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)  # Encode the text features using CLIP model


    evaluate_model(dataloader, model, class_names, text_features, device)

if __name__ == "__main__":
    main()
