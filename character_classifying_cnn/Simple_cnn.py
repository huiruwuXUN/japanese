import argparse
import os
import numpy as np
import torch.optim as optim
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

def image_preprocessing(img_name):
    '''
        Transfer the RGB image into 64*64 Gray scale image
    '''
    img_gray = cv2.imread(img_name, 0)
    img_resized = cv2.resize(img_gray, (64, 64))
    img_resized_normalized = img_resized.astype('float32') / 255.0
    return img_resized_normalized

def dataset(class_names, root_path, if_split=True):
    """
        Preprocess the image and build up the datasets.
    """
    images = []
    labels = []

    folders = []
    for i in range(len(class_names)):
        folders.append((class_names[i], i))

    for folder, label in folders:
        folder_path = os.path.join(root_path, folder)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_processed = image_preprocessing(img_path)
                images.append(img_processed)
                labels.append(label)

    images_np = np.array(images)
    labels_np = np.array(labels)

    n_classes = int(np.max(labels_np) + 1)
    labels_one_hot = np.eye(n_classes)[labels_np]

    images_tensor = torch.tensor(images_np).float().unsqueeze(1)  # Adds a channel dimension
    labels_tensor = torch.tensor(labels_one_hot).float()

    dataset = TensorDataset(images_tensor, labels_tensor)
    batch_size = 64
    if if_split:
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        return train_loader, val_loader, n_classes
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return data_loader, n_classes

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        
        self.fc1 = nn.Linear(32 * 16 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def train(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, torch.max(labels, 1)[1])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}')

    print("Training complete")

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, torch.max(labels, 1)[1])
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == torch.max(labels, 1)[1]).sum().item()

    print(f'Validation Loss: {val_loss / len(val_loader)}')
    print(f'Accuracy: {100 * correct / total}%')

def main(train_model, img_dir, save_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    if train_model:
        train_loader, val_loader, num_classes = dataset(['kana', 'kanji'], img_dir)
        model = SimpleCNN(num_classes).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train(model, train_loader, criterion, optimizer, device, epochs=20)
        validate(model, val_loader, criterion, device)
        torch.save(model.state_dict(), save_dir)

    else:
        val_loader, num_classes = dataset(['kana', 'kanji'], img_dir, if_split=False)
        model = SimpleCNN(num_classes).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.load_state_dict(torch.load(save_dir))
        validate(model, val_loader, criterion, device)


if __name__ == "__main__":
    # args using in the code/
    train_model = False
    image_dir = 'character_classifying_cnn\outputs\images\pilot_set'
    output_dir = 'character_classifying_cnn/outputs/models/model_1.pth'


    main(train_model, image_dir, output_dir)


