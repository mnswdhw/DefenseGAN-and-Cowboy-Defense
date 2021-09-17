# DefenseGAN-and-Cowboy-Defense
This repository implements the two popular defense architectures DefenseGAN and Cowboy that leverage GAN to protect classifiers against test time evasive adversarial attacks <br>

Both of these defense architectures require trained Generator and Discriminator on clean dataset, hence we train a GAN on MNIST dataset. <br>

models.py implements the GAN models in pytorch and directly follows the implementation of the DCgan paper https://arxiv.org/abs/1511.06434. <br>

dc_gan.py takes these models defined in models.py and trains a GAN from scratch in pytorch lightning. The Gan training has been stabilized by employing some of the tricks from the popular repository https://github.com/soumith/ganhacks. <br>

* 1: BatchNorm 
* 2: Label Smoothing 
* 3 : Adding Noise to the inputs of the Discrimnator 

The above hacks helped a lot in stabilizing gan training, earlier while training the gan, the discriminator loss became 0 in very few number of epochs which is a red flag because it meant the discriminator had learnt very quickly and subsequent training would not help the generator very much to improve. <br>

The below image shows the gan training when it was unstable here we can see clearly that the discriminator loss is becoming very small in about 100 epochs. <br>

![Alt Text](/assets/unstable.png)

<br>

The below images shows the gan training after using the stabilizing techniques, it also compares the loss curves with the previous unstable training, here we can observe that discriminator loss is less steeper than before, this results in better quality of trained generator. <br>
![Alt Text](/assets/compare.png)

<br>

The file defense.py implements the DefenseGAN and cowboy defense architectures. <br>
DefenseGAN requires the trained generator to project the adversarial image to the manifold learnt by the generator. This can help in removing the perturbations, for more theoretical insight refer to my presentation in this repository or the paper https://arxiv.org/abs/1805.06605 <br>

![Alt Text](/assets/defense_gan.png)

Cowboy Defense on the other hand uses both the trained generator and the trained discriminator for constructing a defense architecture. The code follows the paper https://arxiv.org/abs/1805.10652 <br>


