import os
from typing import List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class UltrasoundDataset(Dataset):
    """
    Dataset for loading ultrasound images preprocessed for UltraSAM input.
    Assumes images are saved in a directory structure:
        root/train/*.png
        root/val/*.png
        root/test/*.png
    """

    def __init__(self, image_dir: str, transform: Optional[transforms.Compose] = None):
        """
        Args:
            image_dir (str): Directory containing ultrasound images.
            transform (transforms.Compose, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.image_filenames = [f for f in os.listdir(image_dir)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert("RGB")  # Ensure 3 channels

        if self.transform:
            image = self.transform(image)

        return image


def get_transforms():
    """
    Compose preprocessing transforms: ToTensor and normalization, matching UltraSAM input.

    Returns:
        transforms.Compose: Composed torchvision transforms.
    """
    return transforms.Compose([
        transforms.ToTensor(),  # Converts HWC [0,255] PIL image to CHW [0,1] float tensor
        # Optional: add normalization if UltraSAM requires it; use mean/std if known
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
    ])


def create_data_loader(image_dir: str,
                       batch_size: int = 4,
                       shuffle: bool = True,
                       num_workers: int = 4,
                       pin_memory: bool = True) -> DataLoader:
    """
    Utility function to create a DataLoader for ultrasound images.

    Args:
        image_dir (str): Path to dataset directory.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of subprocesses to use for data loading.
        pin_memory (bool): Whether to pin memory (recommended if using GPU).

    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    dataset = UltrasoundDataset(image_dir=image_dir, transform=get_transforms())
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        drop_last=True)
    return loader


if __name__ == "__main__":
    # Example usage and simple test
    import torch

    # Path to your preprocessed + normalized + padded images
    test_data_dir = "data/padded/"

    loader = create_data_loader(test_data_dir, batch_size=2, shuffle=False, num_workers=0)

    for batch_idx, images in enumerate(loader):
        print(f"Batch {batch_idx} - image batch shape: {images.shape}")
        # Expected output: torch.Size([2, 3, 1024, 1024])
        # where 3 is for RGB channels
        if batch_idx == 2:
            break
