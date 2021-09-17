import dc_gan
import torch 
from torch.utils.tensorboard import SummaryWriter
import torchvision

N = 10 
writer = SummaryWriter('defense_logs/version4/')
z = torch.randn(N,100,1,1)
gen = dc_gan.generator1
output = gen(z)


img_grid = torchvision.utils.make_grid(output)
writer.add_image('10_gan_generated_img', img_grid)

