import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import RAVDESSDataset
from model import EmotionCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
dataset = RAVDESSDataset(root_dir="../audio_speech_actors_01-24")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model
model = EmotionCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 15

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for mel, labels in dataloader:
        mel = mel.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(mel)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Loss: {running_loss/len(dataloader):.4f} "
          f"Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), "emotion_model.pth")
print("Training complete. Model saved as emotion_model.pth")
