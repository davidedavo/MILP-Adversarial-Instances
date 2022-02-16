from typing import Optional
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
import numpy as np


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = ".files/", batch_size: int = 32):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.mnist_train, self.mnist_val, self.mnist_test = None, None, None

    def get_data_bounds(self):
        bounds = np.array([[0, 255]], dtype=np.uint8)
        bounds = self.transform(bounds).squeeze()
        return bounds[0].item(), bounds[1].item()

    def prepare_data(self):
        '''called only once and on 1 GPU'''
        # download data (train/val and test sets)
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        print(f"--- SETUP DATAMODULE stage:{stage} ---")
        mnist_full = MNIST(self.data_dir, train=True, download=True, transform=self.transform)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        self.mnist_test = MNIST(self.data_dir, train=False, download=True, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)
