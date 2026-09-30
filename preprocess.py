# preprocess.py
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def load_data(data_dir, batch_size=32, augment=False):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    if augment:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    dataset = datasets.ImageFolder(root=f"{data_dir}", transform=transform)

    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, dataset.classes


def visualize_data(train_loader, classes, num_samples=5):
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    images = images.numpy().transpose(0, 2, 3, 1)
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        axes[i].imshow(np.clip(images[i], 0, 1))
        axes[i].axis('off')
        axes[i].set_title(f"Label: {classes[labels[i]]}")
    plt.show()

def visualize_autoencoder_output(model, val_loader, device):
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(val_loader))
        images = images[:5].to(device)
        reconstructed = model(images)

        images = images.cpu().numpy()
        reconstructed = reconstructed.cpu().numpy()

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        for i in range(5):
            axes[0, i].imshow(np.transpose(images[i], (1, 2, 0)))
            axes[0, i].set_title("Original", fontsize=14)
            axes[0, i].axis('off')

            axes[1, i].imshow(np.transpose(reconstructed[i], (1, 2, 0)))
            axes[1, i].set_title("Reconstructed", fontsize=14)
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.show()


import matplotlib.pyplot as plt

def plot_results(results):
    """
    Plot a bar chart of model names vs accuracy.
    `results`: dict with keys as model names and values as accuracy percentages.
    """
    model_names = list(results.keys())
    accuracies = list(results.values())

    plt.figure(figsize=(12, 6))
    plt.bar(model_names, accuracies, color='lightgreen')
    plt.xlabel('Model / Task')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison of All Models')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


