import numpy as np
import torch
from extract_images import image_extraction_from_file
from dataset import *
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
    
        # Define the encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*10*10, 64),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 64*10*10),
            nn.ReLU(),
            # Reshape to the output of the last conv layer in the encoder
            nn.Unflatten(1, (64, 10, 10)),
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),  
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)
    

def train(device, dataloader, num_epochs=30, lr=1e-4):
    model = AutoEncoder()
    model.to(device)
    # criterion = nn.SmoothL1Loss()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(num_epochs):
        total_loss = 0
        num_images = 0  # 初始化图片总数计数器

        for images in dataloader:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images) * 10
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_images += images.size(0)  # 累加这个批次的图片数量

        avg_loss = total_loss / num_images  # 计算所有批次的平均损失
        print(f'Epoch: {epoch + 1}, average loss: {avg_loss:.6f}')
        losses.append(avg_loss)
    
    plt.figure()
    plt.title('Training Losses curve')
    plt.plot(losses)
    plt.show()

    plt.figure()
    plt.subplot(121)
    plt.imshow(images[0, 0, ...].cpu().detach().numpy(), cmap='gray')
    plt.subplot(122)
    plt.imshow(outputs[0, 0, ...].cpu().detach().numpy(), cmap='gray')
    plt.show()

    
    return model


def main():
    os.chdir('character_classifying_cnn\outputs')
    img_paths = []
    writers = ['B', 'C', 'D', 'Denvour', 'Grant', 'Monica']
    for writer in writers:
        img_paths += image_extraction_from_file(f'class_{writer}.csv', f'images\{writer}', 'Kanji')
    # img_paths = image_extraction_from_file('class_B.csv', 'images\B', 'Kanji')

    dataloader = get_dataloader(image_paths=img_paths, batch_size=32, transform=transform)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train(device, dataloader, num_epochs=100,lr=1e-2)
    print(model)
    model_path = '../../clustering_with_AE/models/model_1.pth'  # 指定保存路径
    torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    main()
