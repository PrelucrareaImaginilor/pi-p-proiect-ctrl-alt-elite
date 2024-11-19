import torch
from torchvision import transforms
from PIL import Image
from main import CNNModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNModel()
model.load_state_dict(torch.load('pancreatic_cancer_model.pth', map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
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
        print(f"Prediction: {prediction}, Confidence: {confidence:.4f}")

predict_image("DATASET/test/normal/1-081.jpg")
