import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

class HandwritingDataset(Dataset):
    def __init__(self, image_dir, transform=None, num_pairs_per_writer=10):
        self.image_dir = image_dir
        self.transform = transform
        self.num_pairs_per_writer = num_pairs_per_writer 
        self.image_list = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.image_dict = self._create_image_dict()

    def _create_image_dict(self):
        # group images by writer ID
        image_dict = {}
        for image_file in self.image_list:
            writer_id = image_file.split('-')[0]  
            if writer_id not in image_dict:
                image_dict[writer_id] = []
            image_dict[writer_id].append(image_file)
        return image_dict

    def __len__(self):
        return len(self.image_list) * self.num_pairs_per_writer

    def __getitem__(self, idx):
        # Select a random writer ID for the anchor and positive
        writer_id = random.choice(list(self.image_dict.keys()))
        if len(self.image_dict[writer_id]) > 1:
            # Randomly choose two images from the same writer to form anchor and positive
            anchor_image, positive_image = random.sample(self.image_dict[writer_id], 2)
        else:
            return self.__getitem__(random.randint(0, len(self) - 1))  # Retry if not enough images for positive pair

        # Select a random writer ID different from the anchor's writer for the negative sample
        negative_writer_id = random.choice([w for w in self.image_dict.keys() if w != writer_id])
        negative_image = random.choice(self.image_dict[negative_writer_id])

        anchor_image = Image.open(os.path.join(self.image_dir, anchor_image))
        positive_image = Image.open(os.path.join(self.image_dir, positive_image))
        negative_image = Image.open(os.path.join(self.image_dir, negative_image))

        # Apply the transform if provided
        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)
        else:
            # Default transform
            transform = transforms.ToTensor()
            anchor_image = transform(anchor_image)
            positive_image = transform(positive_image)
            negative_image = transform(negative_image)

        return anchor_image, positive_image, negative_image



class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 29 * 29, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, anchor, positive, negative):
        # extract features for anchor
        anchor_output = self.cnn(anchor)
        anchor_output = anchor_output.view(anchor_output.size(0), -1)
        anchor_output = self.fc(anchor_output)

        # extract features for positive image
        positive_output = self.cnn(positive)
        positive_output = positive_output.view(positive_output.size(0), -1)
        positive_output = self.fc(positive_output)

        # extract features for negative image
        negative_output = self.cnn(negative)
        negative_output = negative_output.view(negative_output.size(0), -1)
        negative_output = self.fc(negative_output)

        return anchor_output, positive_output, negative_output



class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Compute the distance between anchor and positive, and between anchor and negative
        positive_distance = nn.functional.pairwise_distance(anchor, positive)
        negative_distance = nn.functional.pairwise_distance(anchor, negative)

        # Compute the triplet loss
        loss = torch.clamp(positive_distance - negative_distance + self.margin, min=0.0)

        return torch.mean(loss)

def train():
    start_time = time.time()
    
    # Create dataset and data loaders (train, validation, test)
    train_dataset = HandwritingDataset(image_dir="E:/Data/JHA/CASIA_modified/HWDB2.1Test")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
    print(f'Total number of training samples: {len(train_dataset)}')
    print(f'Total number of training batches: {len(train_loader)}')

    val_dataset = HandwritingDataset(image_dir="E:/Data/JHA/CASIA_modified/HWDB2.0Test")
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)
    print(f'Total number of validation samples: {len(val_dataset)}')
    print(f'Total number of validation batches: {len(val_loader)}')

    early_stopping_patience = 3
    train_losses = []
    val_losses = []

    # Initialize model, loss function, optimizer, and scaler for mixed precision
    model = SiameseNetwork().cuda()
    criterion = TripletLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()  # For mixed precision

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
            anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()

            optimizer.zero_grad()
            
            # Mixed precision training
            with autocast():
                anchor_output, positive_output, negative_output = model(anchor, positive, negative)
                loss = criterion(anchor_output, positive_output, negative_output)

            # Scales the loss, calls backward() and updates weights
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if batch_idx % 100 == 99:
                print(f'Finished {batch_idx+1} / {len(train_loader)} batches')
                current_time = time.time()
                time_used = round((current_time - start_time)/60, 1)
                print(f'Time used: {time_used} minutes')

        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss}')
        train_losses.append(avg_train_loss)

        # Validation phase (same logic as before)
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
                anchor_output, positive_output, negative_output = model(anchor, positive, negative)
                loss = criterion(anchor_output, positive_output, negative_output)
                val_running_loss += loss.item()

        avg_val_loss = val_running_loss / len(val_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss}')
        val_losses.append(avg_val_loss)

        if len(val_losses) > early_stopping_patience:
            if all(val_losses[-i] > val_losses[-i-1] for i in range(1, early_stopping_patience+1)):
                print('Early stopping due to increasing validation loss.')
                break

        torch.save(model.state_dict(), 'triplet_model.pth')


def inference():
    model = SiameseNetwork().cuda() 
    model.load_state_dict(torch.load('siamese_model.pth'))
    model.eval()

    with torch.no_grad():
        img_path_1 = "E:/Data/JHA/CASIA_modified/HWDB2.0Test/020-P16_9.jpeg"
        img_path_2 = "E:/Data/JHA/CASIA_modified/HWDB2.0Test/006-P18_7.jpeg"
        image1 = Image.open(img_path_1)
        image2 = Image.open(img_path_2)
        transform = transforms.ToTensor()
        image1 = transform(image1).unsqueeze(0).cuda()
        image2 = transform(image2).unsqueeze(0).cuda()
        output1, output2 = model(image1, image2)
        distance = nn.functional.pairwise_distance(output1, output2)
        print(f'Comparing image \n{img_path_1} and image \n{img_path_2}')
        print(f'Predicted Distance: {distance.item()}')
        if distance.item() < 0.5:
            print("Same writer")
        else:
            print("Different writer")

def test():
    test_dataset = HandwritingDataset(image_dir="E:/Data/JHA/CASIA_modified/HWDB2.1Test")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    print(f'Total number of testing samples: {len(test_dataset)}')
    print(f'Total number of testing batches: {len(test_loader)}')

    result = np.array([]).reshape(0,2)

    model = SiameseNetwork().cuda()
    model.load_state_dict(torch.load('siamese_model.pth'))
    model.eval()

    with torch.no_grad():
        for image1, image2, label in test_loader:
            image1, image2, label = image1.cuda(), image2.cuda(), label.cuda()
            output1, output2 = model(image1, image2)
            distance = nn.functional.pairwise_distance(output1, output2)
            result = np.concatenate((result, torch.cat((distance.view(-1,1), label.view(-1,1)), dim=1).cpu().numpy()), axis=0)
    plt.scatter(result[:,0], result[:,1])
    plt.show()
    

if __name__ == '__main__':
    train()
    # inference()
    # test()

