from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_fashion_mnist(batch_size=256, root="../data"):
    transform = transforms.ToTensor()

    train_dataset = datasets.FashionMNIST(
        root=root,
        train=True,
        transform=transform,
        download=True
    )

    test_dataset = datasets.FashionMNIST(
        root=root,
        train=False,
        transform=transform,
        download=True
    )

    train_iter = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_iter = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_iter, test_iter