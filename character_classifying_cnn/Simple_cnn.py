import argparse
import os
import random
from matplotlib import pyplot as plt
import numpy as np
import torch.optim as optim
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
from PIL import Image
import csv
import csv

def set_seed(seed_value=42):
    """设置随机数种子以确保实验结果的可重复性"""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # 如果使用多个GPU
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def data_augmentation(img):
    '''
        Data augmentation for the image
    '''
    img = Image.fromarray(img)
    transform = transforms.Compose([
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)), 
    transforms.Resize((64, 64)), 
    transforms.RandomHorizontalFlip(),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),

])

    img_transformed = transform(img)

    img_transformed_array = np.array(img_transformed)

    return img_transformed_array

def image_preprocessing(img_name, if_aug=False):
    '''
        Transfer the RGB image into 64*64 Gray scale image
    '''
    img_gray = cv2.imread(img_name, 0)
    img_resized = cv2.resize(img_gray, (64, 64))
    img_resized_normalized = img_resized.astype('float32') / 255.0
    if if_aug:
        img_resized_normalized = data_augmentation(img_resized_normalized)
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
        # 第一个5x5卷积层替换为两个3x3卷积层
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # 第二个5x5卷积层替换为两个3x3卷积层
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        # 由于增加了一层卷积层但保持了池化层，输出特征图尺寸不变
        self.fc1 = nn.Linear(32 * 16 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
    
def test(model, test_loader, criterion, device):
    model.eval()  # 将模型设置为评估模式
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # 在评估过程中不计算梯度
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, torch.max(labels, 1)[1])
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == torch.max(labels, 1)[1]).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100 * correct / total
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return test_loss


def train(model, train_loader, test_loader, criterion, optimizer, device, epochs=10):
    model.train()
    test_losses = []
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

        test_loss = test(model, test_loader, criterion, device)
        test_losses.append(test_loss)
    plt.figure()
    plt.plot(test_losses)
    plt.xlabel('Training epochs')
    plt.ylabel('Test Loss')
    plt.savefig('character_classifying_cnn\outputs\models\model_3')
    plt.show()


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

def inference(model, folder_path, save_dir, device):
    image_names = []
    images = []
    model.eval()
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_processed = image_preprocessing(img_path)
            image_names.append(img_name)
            images.append(torch.tensor(img_processed).unsqueeze(0).unsqueeze(0).to(device))

    file_name = f'{save_dir}/class_Monica.csv'
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image_name', 'Class'])
        for i in range(len(image_names)):
            label = model(images[i])
            if label[0, 0] > label[0, 1]:
                writer.writerow([image_names[i], 'Kana'])
            else:
                writer.writerow([image_names[i], 'Kanji'])





def main(train_model, valid, to_inference, img_dir, save_dir):
def inference(model, folder_path, save_dir, device):
    image_names = []
    images = []
    model.eval()
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_processed = image_preprocessing(img_path)
            image_names.append(img_name)
            images.append(torch.tensor(img_processed).unsqueeze(0).unsqueeze(0).to(device))

    file_name = f'{save_dir}/class_Monica.csv'
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image_name', 'Class'])
        for i in range(len(image_names)):
            label = model(images[i])
            if label[0, 0] > label[0, 1]:
                writer.writerow([image_names[i], 'Kana'])
            else:
                writer.writerow([image_names[i], 'Kanji'])





def main(train_model, valid, to_inference, img_dir, save_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    if train_model:
        train_loader, val_loader, num_classes = dataset(['kana', 'kanji'], img_dir)
        model = SimpleCNN(num_classes).to(device)

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        train(model, train_loader, val_loader,criterion, optimizer, device, epochs=100)
        # validate(model, val_loader, criterion, device)
        torch.save(model.state_dict(), save_dir)

    elif valid:
    elif valid:
        val_loader, num_classes = dataset(['kana', 'kanji'], img_dir, if_split=False)
        model = SimpleCNN(num_classes).to(device)

        criterion = nn.NLLLoss()
        # optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.load_state_dict(torch.load(save_dir))
        validate(model, val_loader, criterion, device)

    elif to_inference:
        model = SimpleCNN(2).to(device)
        criterion = nn.NLLLoss()
        model.load_state_dict(torch.load(save_dir))
        inference(model, img_dir, 'character_classifying_cnn/outputs', device)
        

    elif to_inference:
        model = SimpleCNN(2).to(device)
        criterion = nn.NLLLoss()
        model.load_state_dict(torch.load(save_dir))
        inference(model, img_dir, 'character_classifying_cnn/outputs', device)
        


if __name__ == "__main__":

    set_seed(42)
    
    train_model = False
    to_inference = True
    test_image_dir = 'character_classifying_cnn\outputs\images\Monica'
    train_image_dir = 'character_classifying_cnn\outputs\images'
    output_dir = 'character_classifying_cnn/outputs/models/model_4.pth'


    main(False, False, to_inference, test_image_dir, output_dir)


