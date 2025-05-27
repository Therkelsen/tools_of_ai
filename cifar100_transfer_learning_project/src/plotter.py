import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as nps

def test_transforms(img, file_path):
    transform1 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    transform2 = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor()
    ])
    
    transform3 = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),  # always flip for demo
        transforms.ToTensor()
    ])

    transform4 = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])

    transform5 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor()
    ])

    # Apply transforms separately to the same image
    img1 = transform1(img)
    img2 = transform2(img)
    img3 = transform3(img)
    img4 = transform4(img)
    img5 = transform5(img)

    # Show original and transformed images side-by-side
    fig, axs = plt.subplots(1, 6, figsize=(15, 5))
    axs[0].imshow(img)
    axs[0].set_title("Original")
    axs[0].axis('off')

    for i, t_img in enumerate([img1, img2, img3, img4, img5]):
        # Convert tensor image back to numpy for plotting (C x H x W to H x W x C)
        t_img_np = t_img.permute(1, 2, 0).numpy()
        axs[i+1].imshow(t_img_np)
        axs[i+1].axis('off')

    axs[1].set_title("Resize")
    axs[2].set_title("Color Jitter")
    axs[3].set_title("Horizontal Flip")
    axs[4].set_title("Random Rotation")
    axs[5].set_title("Combination")

    plt.tight_layout()
    plt.savefig(file_path)

def show_images_per_class(dataset, class_names, selected_classes, file_path, num_images=3):
    # Remove redundant transform
    # transform = transforms.Compose([
    #     transforms.ToTensor()])

    # Map from selected class index (0-9) to original CIFAR class idx (like 49, etc)
    class_to_cifar_label = {i: c for i, c in enumerate(selected_classes)}
    # Also create reverse mapping: cifar label to index in selected_classes
    cifar_label_to_class = {c: i for i, c in enumerate(selected_classes)}

    images_per_class = {i: [] for i in range(len(class_names))}

    for img, label in dataset:
        # img is already a tensor
        if label in cifar_label_to_class:
            class_idx = cifar_label_to_class[label]
            if len(images_per_class[class_idx]) < num_images:
                images_per_class[class_idx].append(img)
            if all(len(imgs) >= num_images for imgs in images_per_class.values()):
                break

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(num_images, len(class_names), figsize=(len(class_names) * 3, num_images * 3))
    for class_idx, imgs in images_per_class.items():
        for img_idx, img in enumerate(imgs):
            ax = axes[img_idx, class_idx] if len(class_names) > 1 else axes[img_idx]
            img = img.permute(1, 2, 0)
            img = img.cpu().numpy()

            ax.imshow(img)
            ax.axis('off')
            if img_idx == 0:
                ax.set_title(f"{class_names[class_idx]}")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def plot_confusion_matrix(model, dataloader, class_names, file_path, model_name, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            outputs = outputs['logits'] if isinstance(outputs, dict) else outputs
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45, cmap='Blues')
    plt.title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CIFAR-100 dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

    # Define selected classes by name
    class_names = ['bicycle', 'bus', 'lawn_mower', 'motorcycle', 'pickup_truck',
                   'rocket', 'streetcar', 'tank', 'tractor', 'train']

    # Map class names to CIFAR-100 class indices
    selected_classes = [dataset.class_to_idx[name] for name in class_names]

    # Filter dataset to only include selected classes
    indices = [i for i, (_, label) in enumerate(dataset) if label in selected_classes]
    subset = Subset(dataset, indices)

    # Show sample images per class
    show_images_per_class(dataset, class_names, selected_classes, "images_per_class.png")

    # Test transformations on a single image
    img, label = subset[5]
    img_pil = transforms.ToPILImage()(img)
    test_transforms(img_pil, "transform_examples.png")

if __name__ == "__main__":
    main()
