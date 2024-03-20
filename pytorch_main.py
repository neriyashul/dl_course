import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision import datasets, transforms
import random
import shutil
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------

'''
TODO: 
move 500 randomly files from train to val using the linux commands:
!find /content/chest_xray/train/NORMAL -type f -print0 | shuf -n 140 -z | xargs -0 mv -t /content/chest_xray/val/NORMAL
!find /content/chest_xray/train/PNEUMONIA -type f -print0 | shuf -n 360 -z | xargs -0 mv -t /content/chest_xray/val/PNEUMONIA
'''
# ---------------------------------------------------------------------


# TODO: delete it
def show_img(image):
    image_np = image.numpy()  # Convert to numpy array
    image_np = np.transpose(image_np, (1, 2, 0))
    # Display the image with matplotlib
    plt.imshow(image_np)
    plt.axis('off')  # Optionally, turn off the axis.
    plt.show()
    
# TODO: delete it
def show_histogram(image):
    # Display the histogram
    plt.hist(np.array(image).ravel(), bins=256) 
    plt.show()

# ---------------------------------------------------------------------




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_dir = "~/content/chest_xray"



# resize the images to 256x256
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Set to grayscale
    transforms.Resize((256, 256))
])

train_dataset = datasets.ImageFolder(root=os.path.join(base_dir, "train/"), transform=transform)
val_dataset = datasets.ImageFolder(root=os.path.join(base_dir, "val/"), transform=transform)
test_dataset = datasets.ImageFolder(root=os.path.join(base_dir, "test/"), transform=transform )


# ---------------------------------------------------------------------
def calculate_mean_std(dataset):
    # total_pixels = np.concatenate([np.array(image).ravel() for image, _ in dataset])
    # mean = np.mean(total_pixels)
    # std = np.std(total_pixels)
    mean_list = []
    std_list = []
    for image, _ in dataset:
        pixels = np.array(image).ravel()
        mean_list.append(np.mean(pixels))
    mean = np.mean(mean_list)
    
    for image, _ in dataset:
        pixels = np.array(image).ravel()
        a = pixels - mean
        std_list.append(np.sqrt(np.mean(a**2))) 
        
        
    std = np.mean(std_list)
    return mean, std
    return mean, std

def normalize(dataset):
    mean, std = calculate_mean_std(dataset)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])
    return datasets.ImageFolder(root=dataset.root, transform=transform)

# normalize the data with mean and std (z-score normalization)
# train_dataset = normalize(train_dataset)
# ---------------------------------------------------------------------

# Create data loaders
# shuffle the data in the training set so the model does not learn the order of the images
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)


# ---------------------------------------------------------------------

# prompt: create a cnn network using pytorch that gets an images from "/content/chest_xray/train/" and check them with validation from "/content/chest_xray/val/" and test from "/content/chest_xray/test/"


# Define the CNN architecture
class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
    self.fc1 = nn.Linear(32 * 7 * 7, 128)
    self.fc2 = nn.Linear(128, 2)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, 2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2)
#    x = x.view(-1, 32 * 7 * 7)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)

# Train the model
model = CNN()

# ---------------------------------------------------------------------

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------------------------------------------------------------
# Train the CNN with images and corresponding labels.
# ---------------------------------------------------------------------

for epoch in range(10):
  for i, (images, labels) in enumerate(train_loader):
    # Move tensors to the configured device
    images = images.to(device)
    labels = labels.to(device)

    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print statistics
    if (i + 1) % 100 == 0:
      print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
          epoch + 1, 10, i + 1, len(train_loader), loss.item()))


# ---------------------------------------------------------------------

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
  for images, labels in val_loader:
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Test Accuracy: {}%'.format((correct / total) * 100))

