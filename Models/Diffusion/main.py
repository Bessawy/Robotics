import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use an interactive backend


def show_image(dataset, num_samples=20, columns=4):
    """
    Plots some samples from the dataset.
    Args:
        dataset: The dataset to sample from.
        num_samples: The number of samples to display.
        columns: The number of columns in the plot.
    """

    plt.figure(figsize=(20, 20))
    for i, (img, _) in enumerate(dataset):  # Unpack (image, label)
        if i >= num_samples:
            break
        plt.subplot(num_samples // columns + 1, columns, i + 1)
        img = transforms.ToPILImage()(img)  # Convert tensor to PIL image
        plt.imshow(img)  # Display the PIL image
        plt.axis('off')  # Hide axes for better visualization
    plt.show()  # Ensure the plot is displayed

if __name__ == "__main__":
    # Apply transformations to convert images to tensors
    transform = transforms.Compose([
        transforms.ToTensor()  # Convert images to tensors
    ])
    # Load CIFAR10 dataset
    dataset = torchvision.datasets.Food101(root='.', download=True, transform=transform)
    show_image(dataset)