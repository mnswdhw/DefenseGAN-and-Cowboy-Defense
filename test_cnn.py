import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import os

from dataset import Datasets
from models import MnistCnn




batch_size = 50 
adv_root_dir = "./adv/"



def clean_test(model,testloader,device):

    model.eval()
    running_corrects = 0.0
    epoch_size = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(testloader):
            
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)

            # statistics
            running_corrects += torch.sum(preds == labels.data)
            
            epoch_size += inputs.size(0)

    running_corrects =  running_corrects.double() / epoch_size

    print('Test  Acc: {:.4f}'.format(running_corrects))



if __name__ == "__main__":

    model = MnistCnn()
    # model_path = "/home/manas/Desktop/projects/sigmared/Code/optdefensegan/checkpoints/mnist_classifier.pth"
    model_path = "/content/gdrive/MyDrive/projects/optdefensegan/checkpoints/mnist_classifier.pth"

    transform_test = transforms.Compose([
    transforms.RandomCrop(28),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),        

    ])

    mnist = Datasets("mnist",transform_test,batch_size,10000)
    test_loader = mnist.load_test()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    clean_test(model,test_loader,device)

    





