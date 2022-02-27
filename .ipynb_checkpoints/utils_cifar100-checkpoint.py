# regular imports
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import pickle

# pytorch related imports
import torch, torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.optim.lr_scheduler import OneCycleLR # we are not going to use as we are only running 4 epochs

# lightning related imports
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping # not used because we are running only 4 epochs
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary


# import CIFAR100DataModule ?, LightningModule, Trainer, seed_everything

class CIFAR100DataModule(pl.LightningDataModule):
    '''
    The pl_bolts module has a subset of popular data modules convenient to call and use (thus we can
    avoid defining DataLoader to load data from the datasets). But we are not using it because there is
    no CIFAR100 DataModule defined in the pl_bolts module. We define a class 'CIFAR100DataModule to
    serve the same purpose as any other dataset subset module defined inside pl_bolts module.
    
    The methods defined below like prepare_data, setup does the preprocessing such as getting the data
    and splitting them into train, validation and test dataset, and the method train_dataloader, 
    val_dataloader and test_dataloader loads the respective dataset in batches.
    
    '''
    def __init__(self, batch_size, data_dir: str = './data'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomRotation(15),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
                                                  ])
        
        self.transform_test = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
                                                  ])

        
    def prepare_data(self):
        # download 
        torchvision.datasets.CIFAR100(self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            cifar_full = torchvision.datasets.CIFAR100(self.data_dir, train=True, transform=self.transform_train)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.cifar_test = torchvision.datasets.CIFAR100(self.data_dir, train=False, transform=self.transform_test)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=16, num_workers=4)

#function to read files present in the Python version of the dataset
def unpickle(file):
    with open(file, 'rb') as fo:
        myDict = pickle.load(fo, encoding='latin1')
    return myDict

# function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))