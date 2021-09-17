import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from models import MnistGenerator,MnistDiscriminator
from dataset import Datasets
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
import torch.nn.functional as F
from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import os
import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint

mcd = 1
discriminator1 = None
generator1 = None




class CheckpointEveryEpoch(pytorch_lightning.Callback):
    def __init__(self, start_epoc, save_path):
        self.start_epoc = start_epoc
        self.file_path = save_path

    def on_epoch_end(self, trainer: pytorch_lightning.Trainer, _):
        """ Check if we should save a checkpoint after every train epoch """
        epoch = trainer.current_epoch
        if ((epoch+1)%20) == 0:
            ckpt_path = f"{self.file_path}_e{epoch}_try_stable.ckpt"
            trainer.save_checkpoint(ckpt_path)
            
            




class DCGAN(LightningModule):

    def __init__(self,latent_dim: int = 100,lr: float = 0.0002,b1: float = 0.5,b2: float = 0.999,batch_size: int = 50,valid_size : int = 10000, **kwargs):
      super().__init__()
      self.save_hyperparameters(args)

      self.latent_dim = latent_dim
      self.lr = lr
      self.b1 = b1
      self.b2 = b2
      self.batch_size = batch_size
      self.valid_size = valid_size
          

      # networks
      self.img_shape = (1, 28, 28)
      self.generator = MnistGenerator()
      self.discriminator = MnistDiscriminator()

      self.validation_z = torch.randn(8, self.latent_dim,1,1)

      self.example_input_array = torch.zeros(2, self.latent_dim,1,1)
        

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        # print(batch_idx)
        imgs, _ = batch
        # print(optimizer_idx,batch_idx)
        # sample noise
        z = torch.randn(imgs.shape[0], self.latent_dim,1,1)
        z = z.type_as(imgs)

        # train discriminator
        if optimizer_idx == 0:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1) * 0.9 #for label smoothing for stabilizing gan training
            valid = valid.type_as(imgs)
            #imgs_shape = (N,1,28,28)
            #adding noise from gaussian/random normal dist
            #with mean = 0 and std = 0.1 to stabilize gan training 
            random_noise_1 = torch.empty((imgs.shape[0],1,28,28)).normal_(0,0.1)
            imgs = imgs + random_noise_1

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)
            fake_imgs = self(z)
            random_noise_2 = torch.empty((imgs.shape[0],1,28,28)).normal_(0,0.1)
            fake_imgs = fake_imgs + random_noise_2

            fake_loss = self.adversarial_loss(
                self.discriminator(fake_imgs.detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            # self.logger.experiment.add_scalar("discriminator batch Loss", d_loss , batch_idx )
            output_dict = {"loss" : d_loss, "net" : "D"}
            # print(output_dict)
            # return d_loss

        # train generator
        elif optimizer_idx == 1:

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1) * 0.9 #for label smoothing 
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            
            # self.logger.experiment.add_scalar("generator batch Loss", g_loss, batch_idx )
            output_dict = {"loss" : g_loss, "net" : "G"}
            # print(output_dict)

        # print(output_dict)
        return output_dict
        # return g_loss

    def configure_optimizers(self):
        lr = self.lr
        b1 = self.b1
        b2 = self.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        # return ({"optimizer":opt_g,"frequency":1},{"optimizer":opt_d,"frequency":1})
        return [opt_d, opt_g], []

    def train_dataloader(self):
        transformation = transforms.Compose([
            transforms.RandomCrop(28),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ])
        mnist = Datasets("mnist",transformation,self.batch_size,self.valid_size)
        (train_loader,_) = mnist.load_train()
        # print(len(train_loader))
        return train_loader

    def val_dataloader(self):
        transformation = transforms.Compose([
            transforms.RandomCrop(28),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ])
        mnist = Datasets("mnist",transformation,self.batch_size,self.valid_size)
        (_,valid_loader) = mnist.load_train()
        # print(len(valid_loader))

        return valid_loader

    def training_epoch_end(self,outputs):
      
        # print(outputs)
        z = self.validation_z.to(self.device)
        outputs_d = []
        outputs_g = []
        for x in outputs:
          for opt in x:
            if opt["net"] == "D":
              outputs_d.append(opt["loss"])
            else :
              outputs_g.append(opt["loss"])
        avg_loss_d = torch.stack([x for x in outputs_d]).mean()
        avg_loss_g = torch.stack([x for x in outputs_g]).mean()

        self.logger.experiment.add_scalar("Average Gen loss", avg_loss_g,self.current_epoch)
        self.logger.experiment.add_scalar("Average Dis loss", avg_loss_d,self.current_epoch)
        #generate images 
        self.generated_imgs = self(z)
    
        # log sampled images
        sample_imgs = self.generated_imgs[:6]
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image('generated_images', grid,self.current_epoch)

        return None


    # def validation_step(self,batch,batch_idx):
    #     # print(batch_idx)
    #     #calculate discriminator loss on validation images every 500 epochs
    #     if self.current_epoch == 500:
    #         (imgs,_) = batch 
    #         valid = torch.ones(imgs.size(0), 1)
    #         valid = valid.type_as(imgs)
    #         real_loss = self.adversarial_loss(self.discriminator(imgs), valid)  
    #         self.logger.experiment.add_scalar("Validation Discriminator loss", real_loss , batch_idx)

    #         return real_loss



def main(args) :
    logger = TensorBoardLogger(save_dir = "lightning_logs")
    model = DCGAN(**vars(args))
    obj = CheckpointEveryEpoch(100, args.save_path)
    # trainer = Trainer(gpus=args.gpus, max_epochs = 100, logger = logger,tpu_cores=1, callbacks=[obj])
    # model = DCGAN.load_from_checkpoint("lightning_logs/default/version_10/checkpoints/epoch=99-step=99999.ckpt")
    trainer = Trainer(gpus=args.gpus, max_epochs = 500, logger = logger,tpu_cores=1, callbacks=[obj],resume_from_checkpoint = "checkpoints/_e459_try_stable.ckpt")
    trainer.fit(model)
    # model = DCGAN.load_from_checkpoint("/content/gdrive/MyDrive/projects/optdefensegan/checkpoints/_e20.ckpt")
    # print(model.discriminator(torch.rand([2,1,28,28])))
    # discriminator = model.discriminator
    # generator = model.generator
    # return (discriminator,generator)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=0, help="number of GPUs")
    parser.add_argument("--batch_size", type=int, default=50, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100,
                         help="dimensionality of the latent space")
    parser.add_argument("--tpu_b", type = bool, default = False, help = "Tpu yes or no")
    parser.add_argument("--valid_size", type = int, default = 10000, help = "Size of validation dataset")
    parser.add_argument("--save_path", default = "./checkpoints/", help = "save checkpoints path" )

    args = parser.parse_args()
    main(args)

else :
    print("hello")
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=0, help="number of GPUs")
    parser.add_argument("--batch_size", type=int, default=50, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100,
                         help="dimensionality of the latent space")
    parser.add_argument("--tpu_b", type = bool, default = False, help = "Tpu yes or no")
    parser.add_argument("--valid_size", type = int, default = 10000, help = "Size of validation dataset")
    parser.add_argument("--save_path", default = "./checkpoints/", help = "save checkpoints path" )

    args = parser.parse_args()
    (discriminator1,generator1) = main(args)
    # print(model)


    

    




#checkpoints
#add logs per epoch that is on epoch end see the site for logging opencv
#saving checkpoints by trainer.save_checkpoints also add this by creating a callback class which inherits from the base call back class and override the onepochend method

#Inferred : We know that the ouputs received by the on_epoch_end has loss batchwise 
#my doubt was that in this case we are having two losses but as we know that the batch is repeated for both discriminator 
#and generator hence for each batch_index we have an array containing the losses of the optimizers in the seq basically
#since each batch has multiple losses attached to it for each batch we have an array so the structure is 
#[[{"loss" : tensor, "net" : "D"},{"loss" : tensor, "net" : "G"}]] the enner array for batch index

