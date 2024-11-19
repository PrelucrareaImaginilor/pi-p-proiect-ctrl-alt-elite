import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#preprocseasare set de date si setari generale
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(root='DATASET/test', transform=transform)
val_dataset = datasets.ImageFolder(root='DATASET/train', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

#arhitectura cnn
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 30 * 30, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x
#functia de antrenare
    def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.float().to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            #validare rezultate
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.float().to(DEVICE)
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    preds = (outputs > 0.5).float()
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, "
                  f"Val Loss: {val_loss / len(val_loader):.4f}, Accuracy: {correct / total:.4f}")
            # Rulează antrenarea
            model.train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)

            # Salvarea modelului
            torch.save(model.state_dict(), 'pancreatic_cancer_model.pth')
