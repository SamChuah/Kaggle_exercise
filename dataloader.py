import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
import torch.nn.functional as F

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def reshape(img, size=[224,224], train=True):

    if train:
        augment = transforms.Compose([
            transforms.Resize(size=224),
            transforms.CenterCrop(size=224)
            ])
    else:
          augment = transforms.Compose([
            transforms.Resize(size=224),
            transforms.CenterCrop(size=224)
          ])

    return augment(img)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open('Images/' + path).convert('RGB')

def one_hot_(c, num_classes):
    c = torch.tensor(c, dtype=torch.int64)
    return F.one_hot(c, num_classes)

def scale_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]

    return transforms.Compose(t_list)

def get_transform(name='imagenet', input_size=None,
                  scale_size=None, normalize=None, augment=True):
    normalize = __imagenet_stats
    input_size = 256
    if augment:
            return inception_color_preproccess(input_size, normalize=normalize)
    else:
            return scale_crop(input_size=input_size,
                              scale_size=scale_size, normalize=normalize)

class myImageFloder(data.Dataset):
    def __init__(self, img, label, training, loader=default_loader, 
    labelling=one_hot_, resize=reshape):
 
        self.img = img
        self.label = label
        self.training = training
        self.loader = loader
        self.one_hot_ = labelling
        self.resize = resize

    def __getitem__(self, index):
        img  = self.img[index]
        label = self.label[index]

        label = label.astype(np.long)
        img = self.loader(img)
        #label = self.one_hot_(label, 120)
        img = self.resize(img, train=self.training)

        if self.training:  
            w, h = img.size
            processed = get_transform(augment=False)  
            img   = processed(img)

            return img, label
        else:
            processed = get_transform(augment=False)  
            img       = processed(img)
            return img, label

    def __len__(self):
        return len(self.img)
