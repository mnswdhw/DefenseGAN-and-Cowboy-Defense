import foolbox as fb #python library to produce adversarial examples
import torch
import torchvision
from torchvision import transforms
import pickle
import os
from models import MnistCnn
from dataset import Datasets



def fgsm_adv_generate(dataset_name,adv_root_folder,model,device,test_loader):


    model.eval()

    root_path_to_store = adv_root_folder + dataset_name 

    if not os.path.exists(root_path_to_store):
      os.mkdir(root_path_to_store)
    
    fmodel = fb.PyTorchModel(model, bounds=(-1,1),device = device)
    attack = fb.attacks.FGSM()

    fgsm_adv = []
    fgsm_index = []
    fgsm_label = []
    sum = 0

    #for index
    count = 0
    for imgs_batch,labels_batch in test_loader:
      imgs_batch = imgs_batch.to(device)
      labels_batch = labels_batch.to(device)
      raw, clipped, is_adv = attack(fmodel, imgs_batch, labels_batch, epsilons=0.3)
      sum = sum + torch.sum(is_adv)
      for i in range(50):
        fgsm_adv.append(clipped[i])
        fgsm_label.append(labels_batch[i])
        fgsm_index.append(count+i)

      count = count + i + 1
    
    with open (root_path_to_store+'/FGSM_indexs.pickle', 'wb') as fp:
        pickle.dump(fgsm_index, fp)

    with open (root_path_to_store+'/FGSM_adv_images.pickle', 'wb') as fp:
        pickle.dump(fgsm_adv, fp)
        
    with open (root_path_to_store+'/FGSM_adv_label.pickle', 'wb') as fp:
        pickle.dump(fgsm_label, fp)

    print("model accuracy on adversarial examples : ", 1 - (sum/len(fgsm_adv)))
    

    









if __name__ == "__main__":

    #can use config 

    
    dataset_name = "mnist"
    batch_size = 50
    valid_size = 10000

    transformation = transforms.Compose([
    transforms.RandomCrop(28),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),        

    ])

    adv_root_folder = "./adv/"
    if not os.path.exists(adv_root_folder):
        os.mkdir(adv_root_folder)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model_path = "/home/manas/Desktop/projects/sigmared/Code/optdefensegan/checkpoints/mnist_classifier.pth"
    model_path = "/content/gdrive/MyDrive/projects/optdefensegan/checkpoints/mnist_classifier.pth"

    model = MnistCnn()
    model.load_state_dict(torch.load(model_path,map_location = torch.device("cpu"))) # if no gpu else simple torch.load(model_path)
    model.to(device)
    mnist = Datasets("mnist",transformation,batch_size,valid_size)
    test_loader = mnist.load_test()

    fgsm_adv_generate(dataset_name,adv_root_folder,model,device,test_loader)





