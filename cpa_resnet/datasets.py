import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Datasets are downloaded once into <repo>/data (see "Datasets" in the top-level README);
# set CPA_DATA to use another folder.
DATA_DIR = os.path.normpath(os.environ.get(
    'CPA_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')))
CIFAR10_ROOT = DATA_DIR                            # contains cifar-10-batches-py/
CIFAR5M_ROOT = os.path.join(DATA_DIR, 'cifar5m')   # contains part0.npz ... part5.npz

class CIFAR5MDataset(Dataset):
    def __init__(self, root_dir, parts=None, transform=None, target_transform=None):
        """
        root_dir: path to folder containing part*.npz
        parts: list of part filenames or indices, e.g. ["part0.npz","part1.npz"], or None to load all
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

        # identify parts
        if parts is None:
            # list all .npz files sorted
            parts = sorted(
                f for f in os.listdir(root_dir) if f.endswith(".npz")
            )
        self.parts = parts

        # load all data into memory (if fits), or you can load lazily per part
        # Here: load into memory
        imgs = []
        labels = []
        for p in self.parts:
            path = os.path.join(root_dir, p)
            data = np.load(path)
            # assume data["images"], data["labels"]
            imgs.append(data["X"])
            labels.append(data["Y"])
        self.images = np.concatenate(imgs, axis=0)  # shape (M, 32,32,3)
        self.labels = np.concatenate(labels, axis=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]  # shape (32,32,3)
        lbl = int(self.labels[idx])

        # convert to PIL / or to tensor
        # PyTorch expects (C, H, W)
        # you can convert
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0  # into float in [0,1]
        img = transforms.ToPILImage()(img)  # convert to PIL Image
        if self.transform:
            img = self.transform(img)

        return img, lbl