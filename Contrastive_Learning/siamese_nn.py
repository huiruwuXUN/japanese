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
        self.num_pairs_per_writer = num_pairs_per_writer  # 每个书写者生成的正样本对数量
        self.image_list = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.image_dict = self._create_image_dict()  # 按书写者ID将图像分组

    def _create_image_dict(self):
        # 按照书写者ID将图像分组
        image_dict = {}
        for image_file in self.image_list:
            writer_id = image_file.split('-')[0]  # 假设文件名的格式为 "writerID_XXX.jpg"
            if writer_id not in image_dict:
                image_dict[writer_id] = []
            image_dict[writer_id].append(image_file)
        return image_dict

    def __len__(self):
        # 返回合理的长度，根据你想要的数量决定
        return len(self.image_list) * self.num_pairs_per_writer

    def __getitem__(self, idx):
        # 随机选择正样本对或负样本对
        if random.random() > 0.5:  # 50% 机会选择正样本对
            writer_id = random.choice(list(self.image_dict.keys()))
            if len(self.image_dict[writer_id]) > 1:
                # 随机选择同一书写者的两个样本，生成正样本对
                image1, image2 = random.sample(self.image_dict[writer_id], 2)
                label = 1  # 正样本对
            else:
                return self.__getitem__(random.randint(0, len(self) - 1))  # 如果书写者样本不足，重试
        else:  # 50% 机会选择负样本对
            writer_id1, writer_id2 = random.sample(list(self.image_dict.keys()), 2)
            image1 = random.choice(self.image_dict[writer_id1])
            image2 = random.choice(self.image_dict[writer_id2])
            label = 0  # 负样本对

        image1 = Image.open(os.path.join(self.image_dir, image1))
        image2 = Image.open(os.path.join(self.image_dir, image2))

        # 如果有transform则应用transform，否则转换为Tensor
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        else:
            # 默认的Tensor转换
            transform = transforms.ToTensor()
            image1 = transform(image1)
            image2 = transform(image2)

        label = torch.tensor(label, dtype=torch.float32)

        return image1, image2, label


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # 定义一个简单的CNN模型
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
        # 将输入维度调整为卷积层输出展平后的大小 256 * 29 * 29 = 215296
        self.fc = nn.Sequential(
            nn.Linear(256 * 29 * 29, 512),  # 修改这里
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, image1, image2):
        # 提取第一个图像的特征
        output1 = self.cnn(image1)
        output1 = output1.view(output1.size(0), -1)  # 展平成一维向量
        output1 = self.fc(output1)

        # 提取第二个图像的特征
        output2 = self.cnn(image2)
        output2 = output2.view(output2.size(0), -1)
        output2 = self.fc(output2)

        return output1, output2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # 计算两个输出的欧氏距离
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)

        # 根据对比损失函数公式计算损失
        loss = label * torch.pow(euclidean_distance, 2) + \
               (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)

        return torch.mean(loss)

def train():
    start_time = time.time()
    # 创建数据集和数据加载器
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

    # 初始化模型和损失函数
    model = SiameseNetwork().cuda()  # 如果有GPU，请使用.cuda()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # scaler = GradScaler()
    # 训练循环
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_idx = 0

        for batch_idx, (image1, image2, label) in enumerate(train_loader):
            image1, image2, label = image1.cuda(), image2.cuda(), label.cuda()

            # 前向传播
            output1, output2 = model(image1, image2)

            # 计算损失
            loss = criterion(output1, output2, label)

            # 反向传播和优化
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
        image1 = Image.open("E:/Data/JHA/CASIA_modified/HWDB2.0Train/004-P16_8.jpeg")
        image2 = Image.open("E:/Data/JHA/CASIA_modified/HWDB2.0Train/004-P17_0.jpeg")
        transform = transforms.ToTensor()
        image1 = transform(image1).unsqueeze(0).cuda()
        image2 = transform(image2).unsqueeze(0).cuda()
        output1, output2 = model(image1, image2)
        distance = nn.functional.pairwise_distance(output1, output2)
        print(f'Predicted Distance: {distance}')
        if distance < 0.5:
            print("Same writer")
        else:
            print("Different writer")

def test():
    test_dataset = HandwritingDataset(image_dir="E:/Data/JHA/CASIA_modified/HWDB2.1Test")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    print(f'Total number of testing samples: {len(test_dataset)}')
    print(f'Total number of testing batches: {len(test_loader)}')

    result = np.array([]).reshape(0,2)

    model = SiameseNetwork().cuda()  # 如果有GPU，请使用.cuda()
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
    # train()
    inference()
    # test()

