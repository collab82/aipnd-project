import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim, tensor
from torch.autograd import Variable

import matplotlib.image as mpimg

from torchvision import datasets, transforms, models
import torchvision.models as models
import torch.nn.functional as F
from collections import OrderedDict
from PIL import Image

import PIL
from PIL import Image
import json
from matplotlib.ticker import FormatStrFormatter

arch = {"vgg16":25088,
        "densenet121":1024,
        "alexnet":9216}

def load_data(where  = "./flowers" ):
        data_dir = where
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'
        
        train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], 
                                                                   [0.229, 0.224, 0.225])])
        
        valid_transforms = transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], 
                                                                    [0.229, 0.224, 0.225])])
        
        test_transforms = transforms.Compose([transforms.Resize(256), 
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], 
                                                                   [0.229, 0.224, 0.225])])
        
        return trainloader, validloader, testloader

def nn_setup(structure='densenet121',dropout=0.5, hidden_layer1 = 512,lr = 0.001, power=gpu):
    
    
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)        
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif structure == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print(" {} is not a valid model.Did you mean vgg16,densenet121,or alexnet?".format(structure))
        
    
        
    for param in model.parameters():
        param.requires_grad = False

        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
            ('dropout',nn.Dropout(dropout)),
            ('inputs', nn.Linear(structures[structure], hidden_layer1)),
            ('relu1', nn.ReLU()),
            ('hidden_layer1', nn.Linear(hidden_layer1, 90)),
            ('relu2',nn.ReLU()),
            ('h_l2',nn.Linear(90,80)),
            ('relu3',nn.ReLU()),
            ('h_l3',nn.Linear(80,102)),
            ('output', nn.LogSoftmax(dim=1))
                          ]))
        
        
        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr )
        model.cuda()
        
        if torch.cuda.is_available() and power = 'gpu':
            model.cuda()

        return model , optimizer ,criterion

