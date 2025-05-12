import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def data_loader(data_path, batch_size=64):
    """
    Loads images from the specified folder and applies transformations.
    Args:
        data_path: Path to the dataset folder.
        batch_size: Number of images per batch.
    Returns:
        DataLoader object for the dataset.
    """
    # Define transformations (resize, convert to tensor, normalize)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to 128x128
        transforms.ToTensor(),         # Convert images to tensors
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    # Load dataset from the folder
    dataset = ImageFolder(root=data_path, transform=transform)

    # Create a DataLoader for batching
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def visualize_batch(dataloader):
    """
    Visualizes a batch of images from the DataLoader.
    Args:
        dataloader: DataLoader object.
    """
    # Get a batch of images and labels
    images, labels = next(iter(dataloader))

    # Plot the images
    plt.figure(figsize=(10, 10))
    for i in range(min(len(images), 16)):  # Display up to 16 images
        plt.subplot(4, 4, i + 1)
        img = images[i].permute(1, 2, 0)  # Convert (C, H, W) to (H, W, C)
        img = img * 0.5 + 0.5  # Denormalize to [0, 1]
        plt.imshow(img)
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Path to the Cars Dataset folder
    data_path = "../../Datasets/Cars Dataset"

    # Load the dataset
    dataloader = data_loader(data_path)

    # Visualize a batch of images
    visualize_batch(dataloader)

    print("Dataset loaded and visualized.")