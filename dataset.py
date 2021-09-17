# this class should implement the handling of the datasets, including preprocessing images and providing dataloaders

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import random_split, DataLoader
import os

from models import MnistCnn
from torchsummary import summary
import copy


class Datasets:

    def __init__(self,dataset_name,transformation,batch_size,valid_size, root_dir="./data"):

        self.root_dir = root_dir
        self.dataset_name = dataset_name 
        self.transformation = transformation
        self.batch_size = batch_size 
        self.valid_size = valid_size


#return train and valid dataloader
    def load_train(self):

        if self.dataset_name == "mnist":
            trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=self.transformation)


        train_data, val_data = random_split(trainset, [len(trainset) - self.valid_size, self.valid_size])

        

        train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size
        )

        val_loader = DataLoader(
            val_data,
            batch_size=self.batch_size
        )

        return (train_loader,val_loader)



    def load_test(self):

        if self.dataset_name == "mnist":
            testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=self.transformation)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False)

        return test_loader


    









