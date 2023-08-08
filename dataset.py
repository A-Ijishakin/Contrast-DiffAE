import os
import nibabel as nib 
import matplotlib.pyplot as plt 
from io import BytesIO
from pathlib import Path
import lmdb 
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms 
import torchvision 
from torchvision.datasets import CIFAR10, LSUNClass
import torch
import pandas as pd
import numpy as np 
import torchvision.transforms.functional as Ftrans
import random 

# load the slices
class ExampleDataset(Dataset):
    """ SliceLoader 
    
    A class which is used to allow for efficient data loading of the training data. 
    Args:
        - torch.utils.data.Dataset: A PyTorch module from which this class inherits which allows it 
        to make use of the PyTorch dataloader functionalities. 
    
    """
    def __init__(self, train=False, val=False, test=False, epoch_end=True, N=4632):
        #4632
        """ Class constructor
        
        Args:
            - downsampling_factor: The factor by which the loaded data has been downsampled. 
            - N: The length of the dataset. 
            - folder_name: The folder from which the data comes from 
            - is_train: Whether or not the dataloader is loading training data (and therefore randomised data).   
        """ 
        self.val = val 
        self.test = test 
        self.train = train
        self.N = N - 1
        self.data = pd.read_csv('brain_age.csv')
        self.train_data = self.data.iloc[:N] 
        self.val_data = self.data.iloc[4631:]
        self.epoch_end = epoch_end
        
        
    def __len__(self):
        """ __len__
        
        A function which configures and returns the size of the datset. 
        
        Output: 
            - N: The size of the dataset. 
        """
        return (self.N)    
    
    def __getitem__(self, idx):
        transforms_list = [transforms.ToPILImage(), 
                          transforms.RandomAffine(degrees=(0, 350), translate=(0.1, 0.12)), transforms.Resize((64, 64)),
                          transforms.ToTensor()]
        if self.train: 
            #get the filepath
            file_path = self.train_data['file_path'].iloc[idx] 
            #get the root directory of file path
            dataset = file_path.split('/')[0] 
            #load the image
            image = self._load_nib(file_path)[0, :, :, 72] if dataset == 'NACC' else self._load_nib(file_path) 
            #load the label 
            label = random.choice([0, 1])  
            #apply the transforms 
            image = transforms.Compose(transforms_list)(image) if not self.epoch_end else transforms.Compose([transforms_list[0], transforms_list[-2], transforms_list[-1]])(image)

        elif self.val: 
            #get the filepath
            file_path = self.val_data['file_path'].iloc[idx] 
            #get the root directory of file path
            dataset = file_path.split('/')[0] 
            #load the image
            image = self._load_nib(file_path)[0, :, :, 72] if dataset == 'NACC' else self._load_nib(file_path)
            #load the label 
            label = random.choice([0, 1])   
            #apply the transforms 
            image = transforms.Compose([transforms_list[0], transforms_list[-2], transforms_list[-1]])(image)
        

        return {'img': image, 'index': idx, 'label': label, 'file_path': file_path}    
    
    
    def _load_nib(self, filename): 
        """ _load_nib 
        
        A function to load compressed nifti images.
        Args:
            - filename: The name of the file to be loaded. 
        Ouput:
            - The corresponding image as a PyTorch tensor. 
        
        """
        return torch.tensor(nib.load(filename).get_fdata(), dtype=torch.float) 

