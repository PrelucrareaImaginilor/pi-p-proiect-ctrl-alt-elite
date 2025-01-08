import torch
from torchvision import transforms
from PIL import Image
from main import CNNModel
import re

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNModel()
model.load_state_dict(torch.load('pancreatic_cancer_model.pth', map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def predict_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image).item()
        prediction = "Pancreatic Tumor" if output > 0.5 else "Normal"
        confidence = output if output > 0.5 else 1 - output
        im = re.search(r'[^/]+$', image_path)
        print(im.group(0) + " - " + f"Prediction: {prediction}, Confidence: {confidence:.4f}")


for i in range(40, 220):
    predict_image("DATASET/train/normal/1-"+str(f"{i:03}")+".jpg")
