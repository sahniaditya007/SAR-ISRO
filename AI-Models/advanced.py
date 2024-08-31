import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from torch.cuda.amp import GradScaler, autocast

# Define the Depth Estimation Model
class DepthEstimationModel(nn.Module):
    def __init__(self):
        super(DepthEstimationModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1_input_dim = self._get_fc1_input_dim()
        self.fc1 = nn.Linear(self.fc1_input_dim, 512)
        self.fc2 = nn.Linear(512, 3 * 256 * 256)  # Output size matches target size

    def _get_fc1_input_dim(self):
        x = torch.randn(1, 3, 256, 256)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        return int(np.prod(x.size()))

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(x.size(0), 3, 256, 256)  # Reshape to match target size
        return x

# Custom Dataset class
class CustomDepthDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.image_paths = [os.path.join(data_folder, fname) for fname in os.listdir(data_folder) if fname.endswith('.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image  # Using the image as target for demonstration

# Setup the GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model and move to GPU
model = DepthEstimationModel().to(device)

# Define dataset and dataloader with reduced batch size
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

data_folder = r'C:\dataset'
dataset = CustomDepthDataset(data_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # Reduced batch size

# Define loss function and optimizer
criterion = nn.MSELoss()  # Appropriate for regression tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Mixed precision scaler
scaler = GradScaler()

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            depth_outputs = model(images)
            loss = criterion(depth_outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Save the model
torch.save(model.state_dict(), r'C:\Ai Model\depth_estimation_model.pth')

print("Training completed and model saved.")
