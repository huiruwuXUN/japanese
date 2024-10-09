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
        # choose a random writer ID
        if random.random() > 0.5:  # 50% chance to choose a positive sample
            writer_id = random.choice(list(self.image_dict.keys()))
            if len(self.image_dict[writer_id]) > 1:
                # randomly choose two images from the same writer to form a positive pair
                image1, image2 = random.sample(self.image_dict[writer_id], 2)
                label = 1  # positive pair
            else:
                return self.__getitem__(random.randint(0, len(self) - 1))  # retry if the negative sample is not enough
        else:  # 50% chance to choose a negative sample
            writer_id1, writer_id2 = random.sample(list(self.image_dict.keys()), 2)
            image1 = random.choice(self.image_dict[writer_id1])
            image2 = random.choice(self.image_dict[writer_id2])
            label = 0  # negative pair

        image1 = Image.open(os.path.join(self.image_dir, image1))
        image2 = Image.open(os.path.join(self.image_dir, image2))

        # use the transform if it is provided, otherwise use the default transform
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        else:
            # default transform
            transform = transforms.ToTensor()
            image1 = transform(image1)
            image2 = transform(image2)

        label = torch.tensor(label, dtype=torch.float32)

        return image1, image2, label


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

    def forward(self, image1, image2):
        # extract features for the first image
        output1 = self.cnn(image1)
        output1 = output1.view(output1.size(0), -1)
        output1 = self.fc(output1)

        # extract features for the second image
        output2 = self.cnn(image2)
        output2 = output2.view(output2.size(0), -1)
        output2 = self.fc(output2)

        return output1, output2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # compute the Euclidean distance between the two outputs
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)

        # compute the contrastive loss
        loss = label * torch.pow(euclidean_distance, 2) + \
               (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)

        return torch.mean(loss)

def train():
    start_time = time.time()
    # create dataset and data loader
    train_dataset = HandwritingDataset(image_dir="E:/Data/JHA/CASIA_modified/HWDB2.0Train")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print(f'Total number of training samples: {len(train_dataset)}')
    print(f'Total number of training batches: {len(train_loader)}')

    test_dataset = HandwritingDataset(image_dir="E:/Data/JHA/CASIA_modified/HWDB2.0Test")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    print(f'Total number of testing samples: {len(test_dataset)}')
    print(f'Total number of testing batches: {len(test_loader)}')

    early_stopping_patience = 3
    losses = []

    # initialize model, loss function, optimizer
    model = SiameseNetwork().cuda()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # scaler = GradScaler()
    # train the model
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_idx = 0

        for batch_idx, (image1, image2, label) in enumerate(train_loader):
            image1, image2, label = image1.cuda(), image2.cuda(), label.cuda()

            output1, output2 = model(image1, image2)

            loss = criterion(output1, output2, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 99:
                print(f'finished {batch_idx+1} / {len(train_loader)} batches')
                current_time = time.time()
                time_used = round((current_time - start_time)/60,1)
                print(f'time used: {time_used} minutes')
            batch_idx += 1

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')
        losses.append(running_loss/len(train_loader))
        if len(losses) > 3:
            if losses[-1] > losses[-2] and losses[-2] > losses[-3]:
                break


    torch.save(model.state_dict(), 'siamese_model.pth')

    model.eval()
    with torch.no_grad():
        for image1, image2, label in test_loader:
            image1, image2, label = image1.cuda(), image2.cuda(), label.cuda()
            output1, output2 = model(image1, image2)
            distance = nn.functional.pairwise_distance(output1, output2)
            print(f'Predicted Distance: {distance}, Label: {label}')

def inference():
    model = SiameseNetwork().cuda()  # 如果有GPU，请使用.cuda()
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

