import dc_gan
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import math
import os
from dataset import Datasets
from tqdm import tqdm
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import Dataset

# from dataset import Adversarial_Dataset
# from util_defense_GAN import adjust_lr, get_z_sets, get_z_star, Resize_Image
from models import MnistCnn
from torchsummary import summary
import copy
import pickle 

device = "cuda" if torch.cuda.is_available() else "cpu"
writer = SummaryWriter('defense_logs/sigma-variation/')

class Adversarial_Dataset(Dataset):
    
    def __init__(self,transform = None):
      self.transform = transform
      image_name = "/content/gdrive/MyDrive/projects/optdefensegan/adv/mnist/FGSM_adv_images.pickle"
      label_name = "/content/gdrive/MyDrive/projects/optdefensegan/adv/mnist/FGSM_adv_label.pickle"

      with open (image_name, 'rb') as fp:
          self.images = pickle.load(fp)
          
      with open (label_name, 'rb') as fp:
          self.labels = pickle.load(fp)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        
        image = self.images[index].float()
        label = self.labels[index]
        
        if self.transform is not None:
            image = self.transform(image)
        
        return (image, label)








#DEFINE PARAMS
#in official defense gan paper rec_iter = 200 and rec_rr = 10
generator_input_size = 28
learning_rate = 10.0
rec_iter = 200
rec_rr = 10
loss = nn.MSELoss()
INPUT_LATENT = 100 
global_step = 3.0
display_steps = 20
batch_size = 100


def load_adv_dataset():
  sample = Adversarial_Dataset()
  adversarial_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),
  ])
  sample = Adversarial_Dataset(transform = adversarial_transform)

  test_loader = DataLoader(
      sample,
      batch_size=batch_size,
      num_workers=4
  )

  return test_loader

def load_Cnn():
  model = MnistCnn()
  model.load_state_dict(torch.load("/content/gdrive/MyDrive/projects/optdefensegan/checkpoints/mnist_classifier.pth"))
  model = model.to(device)  
  return model

def load_gan():
  dis = dc_gan.discriminator1
  gen = dc_gan.generator1
  dis = dis.to(device)
  gen = gen.to(device)
  return (dis,gen)

def load_test():

  transformation = transforms.Compose([
  transforms.RandomCrop(28),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize((0.5), (0.5)),
  ])

  mnist = Datasets("mnist",transformation,100,10000)
  test_loader = mnist.load_test()

  return test_loader


def defense_on_adversarial_images(sigma,is_defense_gan = False):
  model = load_Cnn()
  (ModelD,ModelG) = load_gan()
  test_loader = load_adv_dataset()

  model.eval()

  running_corrects = 0
  epoch_size = 0


  for batch_idx, (inputs, labels) in tqdm(enumerate(test_loader)):

        data = inputs.to(device)

    # find z*

    if is_defense_gan == True:
      _, z_sets = get_z_sets_defensegan(ModelG,ModelD, data, learning_rate, \
                                  loss, device, rec_iter = rec_iter, \
                                  rec_rr = rec_rr, input_latent = INPUT_LATENT, global_step = global_step)    
    else:    
      _, z_sets = get_z_sets_cowboy(ModelG,ModelD, data, learning_rate, \
                                  loss, device, rec_iter = rec_iter, \
                                  rec_rr = rec_rr, input_latent = INPUT_LATENT, global_step = global_step,sigma = sigma)

    z_star = get_z_star(ModelG, data, z_sets, loss, device)

    # generate data

    data_hat = ModelG(z_star.to(device)).cpu().detach()

    # size back

    # classifier 
    data_hat = data_hat.to(device)

    labels = labels.to(device)

    # evaluate 

    outputs = model(data_hat)
    recon_imgs = data_hat[:5]
    img_grid = torchvision.utils.make_grid(recon_imgs)
    writer.add_image('five_recon_imgs', img_grid)
    img_grid = torchvision.utils.make_grid(inputs[:5])
    writer.add_image('five_adv_perturbed_imgs_fgsm_inf', img_grid)

    _, preds = torch.max(outputs, 1)

    # statistics
    running_corrects += torch.sum(preds == labels.data)
    epoch_size += inputs.size(0)
    

    if batch_idx % display_steps == 0:
        print('{:>3}/{:>3} average acc {:.4f}\r'\
              .format(batch_idx+1, len(test_loader), running_corrects.double() / epoch_size))

    del labels, outputs, preds, data, data_hat,z_star

  test_acc = running_corrects.double() / epoch_size
  print('rec_iter : {}, rec_rr : {}, Test Acc: {:.4f}'.format(rec_iter, rec_rr, test_acc))
  # save_test_results.append(test_acc)
  return test_acc


def defense_on_clean_img(sigma, is_defense_gan = False):
  #LOAD STUFF
  model = load_Cnn()
  (ModelD,ModelG) = load_gan()
  test_loader = load_test()


  model.eval()
  running_corrects = 0
  epoch_size = 0
  # save_test_results = [] 

  

          
  for batch_idx, (inputs, labels) in tqdm(enumerate(test_loader)):

      
      data = inputs.to(device)

      # find z*
      if is_defense_gan == True:
        _, z_sets = get_z_sets_defensegan(ModelG,ModelD, data, learning_rate, \
                                    loss, device, rec_iter = rec_iter, \
                                    rec_rr = rec_rr, input_latent = INPUT_LATENT, global_step = global_step)    
      else:    
        _, z_sets = get_z_sets_cowboy(ModelG,ModelD, data, learning_rate, \
                                    loss, device, rec_iter = rec_iter, \
                                    rec_rr = rec_rr, input_latent = INPUT_LATENT, global_step = global_step,sigma = sigma)

      z_star = get_z_star(ModelG, data, z_sets, loss, device)

      # generate data

      data_hat = ModelG(z_star.to(device)).cpu().detach()

      # size back

      # classifier 
      data_hat = data_hat.to(device)

      labels = labels.to(device)

      # evaluate 

      outputs = model(data_hat)
      recon_imgs = data_hat[:5]
      img_grid = torchvision.utils.make_grid(recon_imgs)
      writer.add_image('five_recon_imgs', img_grid)
      img_grid = torchvision.utils.make_grid(inputs[:5])
      writer.add_image('five_clean_test_imgs', img_grid)
      _, preds = torch.max(outputs, 1)

      # statistics
      running_corrects += torch.sum(preds == labels.data)
      epoch_size += inputs.size(0)

      if batch_idx % display_steps == 0:
          print('{:>3}/{:>3} average acc {:.4f}\r'\
                .format(batch_idx+1, len(test_loader), running_corrects.double() / epoch_size))

      del labels, outputs, preds, data, data_hat,z_star

 
  test_acc = running_corrects.double() / epoch_size
  print('rec_iter : {}, rec_rr : {}, Test Acc: {:.4f}'.format(rec_iter, rec_rr, test_acc))
  
  # save_test_results.append(test_acc)
  return test_acc


#use tensorboard, also save the batch of adversarial image and the corresponding reconstructed image





#UTILITY FUNCTIONS



def get_z_sets_cowboy(modelG,modelD, data, lr, loss, device, rec_iter = 500, rec_rr = 10, input_latent = 100, global_step = 1,sigma = 0.1):
    
    # data = (N,1,28,28)
    display_steps = 100
    
    # the output of R random different initializations of z from L steps of GD
    z_hats_recs = torch.Tensor(rec_rr, data.size(0), input_latent,1,1)
    # shape = (10,N,100)
    
    # the R random differernt initializations of z before L steps of GD
    z_hats_orig = torch.Tensor(rec_rr, data.size(0), input_latent,1,1)
    
    # sigma = 0.1 #hyper param the variance 
    #round prevent overflow

    if sigma < 1 :
      sigma = ((1/sigma) ** 2)/2
    else:
      sigma = (1/(2*(sigma ** 2)))

    # print(sigma)

    #len(z_hats_recs) = 10 
    for idx in range(len(z_hats_recs)):
        
        z_hat = torch.randn(data.size(0), input_latent,1,1).to(device)
        z_hat = z_hat.detach().requires_grad_()
        
        cur_lr = lr

        optimizer = optim.SGD([z_hat], lr = cur_lr, momentum = 0.7)
        
        z_hats_orig[idx] = z_hat.cpu().detach().clone()
        
        # the main optimization loop 
        for iteration in range(rec_iter):
            
            optimizer.zero_grad()
            
            fake_image = modelG(z_hat)
            
            fake_image = fake_image.view(-1, data.size(1), data.size(2), data.size(3))
            
            #change here add another loss term
            loss2 = torch.mean(torch.log(modelD(fake_image)))
            loss1 = sigma * loss(fake_image, data)
            reconstruct_loss = loss1 - loss2
            # print(reconstruct_loss)
            # reconstruct_loss.requires_grad_()
            reconstruct_loss.backward()
            
            optimizer.step()
            
            cur_lr = adjust_lr(optimizer, cur_lr, global_step = global_step, rec_iter= rec_iter)
           
        z_hats_recs[idx] = z_hat.cpu().detach().clone()
        
    return z_hats_orig, z_hats_recs



def get_z_sets_defensegan(modelG,modelD, data, lr, loss, device, rec_iter = 500, rec_rr = 10, input_latent = 100, global_step = 1):
    
    # data = (N,1,28,28)
    display_steps = 100
    
    # the output of R random different initializations of z from L steps of GD
    z_hats_recs = torch.Tensor(rec_rr, data.size(0), input_latent,1,1)
    # shape = (10,N,100)
    
    # the R random differernt initializations of z before L steps of GD
    z_hats_orig = torch.Tensor(rec_rr, data.size(0), input_latent,1,1)

    #len(z_hats_recs) = 10 
    for idx in range(len(z_hats_recs)):
        
        z_hat = torch.randn(data.size(0), input_latent,1,1).to(device)
        z_hat = z_hat.detach().requires_grad_()
        
        cur_lr = lr

        optimizer = optim.SGD([z_hat], lr = cur_lr, momentum = 0.7)
        
        z_hats_orig[idx] = z_hat.cpu().detach().clone()
        
        # the main optimization loop 
        for iteration in range(rec_iter):
            
            optimizer.zero_grad()
            
            fake_image = modelG(z_hat)
            
            fake_image = fake_image.view(-1, data.size(1), data.size(2), data.size(3))
            
            #change here add another loss term
            # loss2 = torch.mean(torch.log(modelD(fake_image)))
            # sigma = 0.1 #hyper param the variance 
            # sigma = 1/(2*(sigma ** 2))
            # loss1 = sigma * loss(fake_image, data)
            reconstruct_loss = loss(fake_image,data)
            # print(reconstruct_loss)
            # reconstruct_loss.requires_grad_()
            reconstruct_loss.backward()
            
            optimizer.step()
            
            cur_lr = adjust_lr(optimizer, cur_lr, global_step = global_step, rec_iter= rec_iter)
           
        z_hats_recs[idx] = z_hat.cpu().detach().clone()
        
    return z_hats_orig, z_hats_recs



"""
To get z* so as to minimize reconstruction error between generator G and an image x
"""

def get_z_star(model, data, z_hats_recs, loss, device):
    
    reconstructions = torch.Tensor(len(z_hats_recs))
    
    for i in range(len(z_hats_recs)):
        
        z = model(z_hats_recs[i].to(device))
        
        z = z.view(-1, data.size(1), data.size(2), data.size(3))
        
        reconstructions[i] = loss(z, data).cpu().item()
        
    min_idx = torch.argmin(reconstructions)
    
    return z_hats_recs[min_idx]


def Resize_Image(target_shape, images):
    
    batch_size, channel, width, height = target_shape
    
    Resize = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((width,height)),
        transforms.ToTensor(),
    ])
    
    result = torch.zeros((batch_size, channel, width, height), dtype=torch.float)
    
    for idx in range(len(result)):
        result[idx] = Resize(images.data[idx])



def adjust_lr(optimizer, cur_lr, decay_rate = 0.1, global_step = 1, rec_iter = 200):
    
    lr = cur_lr * decay_rate ** (global_step / int(math.ceil(rec_iter * 0.032))) # earlier it was 0.8 but 0.8 * 200 = 0.032 * 5000
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    return lr



if __name__ == "__main__":

  # defense_on_clean_img(is_defense_gan = True)
  # defense_on_adversarial_images(is_defense_gan = True)

  sigma = [0.01,0.1,1,10,100]

  #rec_iter = 200, rec_rr = 10
  for sig in sigma:
    test_accuracy_clean = defense_on_clean_img(sig,is_defense_gan = False)
    test_accuracy_adv = defense_on_adversarial_images(sig,is_defense_gan = False)
    writer.add_scalar("Variation of clean test-acc with sigma on cowboy defense",test_accuracy_clean,sig)
    writer.add_scalar("Variation of adv test-acc with sigma on cowboy defense",test_accuracy_adv,sig)

  


  




  





