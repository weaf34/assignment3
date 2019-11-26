from dataloader.food101smallloader import HotDogDataSetLoader
from torchvision.utils import make_grid
from torchvision.transforms import (ToPILImage, ToTensor)
from utils import get_data101 as get_data
from configs import train_path101, label_101_dict
from PIL import Image
import torch
import random

def show_random_data(nrow=4, ncol=4):
    traindata = get_data(train_path101)
    # Collect random samples of each class
    classes = {label_101_dict[name]:
               [path for path,label in traindata if label==label_101_dict[name]]
               for name in label_101_dict}

    samples = {label: random.sample(classes[label], nrow * ncol) for label in classes}

    samples = {label: [*map(lambda x:Image.open(x), samples[label])] for label in classes}

    for label in samples:

        x = [*map(lambda x:x.resize((128,128)),samples[label])]

        x = torch.cat([ToTensor()(x_).unsqueeze(0) for x_ in x],0)

        xgrid = ToPILImage()(make_grid(x,nrow=nrow))

        xgrid.show()

def showrandom_batch():
    dl = HotDogDataSetLoader()
    data = next(iter(dl.train()))
    ToPILImage()(make_grid(data['image'], nrow=5)).show()
