from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#测试用
if __name__ == "__main__":
    from config import dataset, dataset_path, train_batch_size, inference_batch_size
else:
    from .config import dataset, dataset_path, train_batch_size, inference_batch_size

def get_dataloaders():
    if dataset != 'MNIST':
        raise ValueError(f"Unsupported dataset: {dataset}. Only 'MNIST' is supported.")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True}

    train_dataset = MNIST(dataset_path, transform=transform, train=True, download=True)
    test_dataset = MNIST(dataset_path, transform=transform, train=False, download=True)

    print(f"Train dataset size: {len(train_dataset)} samples")
    print(f"Test dataset size: {len(test_dataset)} samples")

    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(dataset=test_dataset, batch_size=inference_batch_size, shuffle=False, **kwargs)
    
    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, test_loader = get_dataloaders()
    print("Data loaders created successfully!")
    print("Train loader batch size:", train_loader.batch_size)
    print("Test loader batch size:", test_loader.batch_size)
    
    try:
        data, labels = next(iter(train_loader))
        print("Successfully fetched a batch from train_loader.")
        print("Data shape:", data.shape)
        print("Labels shape:", labels.shape)

        # Display a sample image
        plt.imshow(data[0].squeeze(), cmap="gray")
        plt.title(f"Label: {labels[0]}")
        plt.show()
        print("Displayed one sample image.")
        
    except Exception as e:
        print("Error fetching a batch:", e)