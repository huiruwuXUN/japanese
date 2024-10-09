import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

class HandwritingDataset(Dataset):
    def __init__(self, npy_file, transform=None, num_pairs_per_writer=10):
        """
        Dataset for handwriting samples, providing triplets of images (anchor, positive, negative).
        
        Parameters:
        npy_file (str): Path to the .npy file containing the entire preprocessed dataset.
        transform (callable, optional): Optional transform to be applied on a sample.
        num_pairs_per_writer (int): Number of triplets to generate for each writer.
        """
        # Load preprocessed dataset from the .npy file
        data_dict = np.load(npy_file, allow_pickle=True).item()
        self.data = data_dict['data']  # Shape: (num_samples, 32, 32)
        self.labels = data_dict['labels']  # Shape: (num_samples,)
        
        self.transform = transform
        self.num_pairs_per_writer = num_pairs_per_writer
        
        # Create a dictionary mapping writer IDs to the indices of their images
        self.image_dict = self._create_image_dict()

    def _create_image_dict(self):
        """
        Create a dictionary where the keys are writer IDs, and the values are lists of image indices.
        """
        image_dict = {}
        for idx, writer_id in enumerate(self.labels):
            if writer_id not in image_dict:
                image_dict[writer_id] = []
            image_dict[writer_id].append(idx)
        return image_dict

    def __len__(self):
        # The length is defined as the number of pairs per writer times the number of data samples
        return len(self.data) * self.num_pairs_per_writer

    def __getitem__(self, idx):
        """
        Generate a triplet: anchor, positive (same writer), and negative (different writer) images.
        """
        # Randomly select a writer ID
        writer_id = np.random.choice(list(self.image_dict.keys()))
        
        # If the writer has more than one image, select two distinct images (anchor and positive)
        if len(self.image_dict[writer_id]) > 1:
            anchor_idx, positive_idx = np.random.choice(self.image_dict[writer_id], 2, replace=False)
        else:
            # Retry if the selected writer has only one image
            return self.__getitem__(np.random.randint(0, len(self)))

        # Select a different writer for the negative sample
        negative_writer_id = np.random.choice([w for w in self.image_dict.keys() if w != writer_id])
        negative_idx = np.random.choice(self.image_dict[negative_writer_id])

        # Load the preprocessed image data
        anchor_image = self.data[anchor_idx]
        positive_image = self.data[positive_idx]
        negative_image = self.data[negative_idx]

        # Convert numpy arrays to tensors and add a channel dimension (grayscale image)
        anchor_image = torch.tensor(anchor_image, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 32, 32)
        positive_image = torch.tensor(positive_image, dtype=torch.float32).unsqueeze(0)
        negative_image = torch.tensor(negative_image, dtype=torch.float32).unsqueeze(0)

        # Apply transformations if provided
        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image



class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5),  # 32x32，28x28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(64, 128, kernel_size=5),  # 10x10
            nn.ReLU(),
            nn.MaxPool2d(2),  # 5x5
            nn.Conv2d(128, 256, kernel_size=3),  # 3x3
            nn.ReLU(),
            nn.MaxPool2d(2)  # 1x1
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 1 * 1, 512),  # 256 * 1 * 1
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)  # 128
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
    
    train_dataset = HandwritingDataset(npy_file="E:/Data/JHA/CASIA_char_imgs_preprocessed.npy", num_pairs_per_writer=5)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=16, pin_memory=True)
    print(f'Total number of training samples: {len(train_dataset)}')
    print(f'Total number of training batches: {len(train_loader)}')

    val_dataset = HandwritingDataset(npy_file="E:/Data/JHA/CASIA_char_imgs_preprocessed_test.npy", num_pairs_per_writer=5)
    val_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=16, pin_memory=True)
    print(f'Total number of validation samples: {len(val_dataset)}')
    print(f'Total number of validation batches: {len(val_loader)}')

    early_stopping_patience = 3
    train_losses = []
    val_losses = []

    # Initialize model, loss function, optimizer, and scaler for mixed precision
    model = SiameseNetwork().cuda()
    model.load_state_dict(torch.load('siamese_model.pth'))
    criterion = TripletLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()  # For mixed precision

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
            anchor = anchor.cuda(non_blocking=True)
            positive = positive.cuda(non_blocking=True)
            negative = negative.cuda(non_blocking=True)

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

            if batch_idx % 1000 == 999:
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
    # 初始化模型并加载预训练权重
    model = SiameseNetwork().cuda()  
    model.load_state_dict(torch.load('triplet_model.pth'))  
    model.eval()

    with torch.no_grad():
        # image paths
        img_path_1 = "E:/Data/JHA/CASIA_char_imgs/Gnt1.0TrainPart2/139-f/00276.png"
        img_path_2 = "E:/Data/JHA/CASIA_char_imgs/Gnt1.0TrainPart2/139-f/00278.png"
        
        # open and preprocess images
        image1 = Image.open(img_path_1).convert("L").resize((32, 32)) 
        image2 = Image.open(img_path_2).convert("L").resize((32, 32))
        
        # Use the same transformations as during training
        transform = transforms.ToTensor()
        image1_tensor = transform(image1).unsqueeze(0).cuda()  # (1, 1, 32, 32)
        image2_tensor = transform(image2).unsqueeze(0).cuda()

        # Extract features from the images using the model
        output1, _, _ = model(image1_tensor, image1_tensor, image1_tensor)  
        output2, _, _ = model(image2_tensor, image2_tensor, image2_tensor)

        # Compute the distance between the two outputs
        distance = nn.functional.pairwise_distance(output1, output2)
        
        # Calculate similarity (inverse of distance) and set conclusion
        similarity = 1 / distance.item()
        threshold = 0.015 
        conclusion = "Same writer" if distance.item() < threshold else "Different writer"
        
        # print similarity and conclusion
        print(f'Comparing image \n{img_path_1} and image \n{img_path_2}')
        print(f'Predicted Distance: {distance.item()}')
        print(f'Similarity: {similarity}')
        print(f'Conclusion: {conclusion}')
        
        # Display the images and similarity
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(image1, cmap='gray')
        plt.title("Image 1")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(image2, cmap='gray')
        plt.title("Image 2")
        plt.axis('off')

        plt.suptitle(f"Similarity: {similarity:.4f} | {conclusion}", fontsize=16)
        plt.show()
        

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
    # train()
    inference()
    # test()

