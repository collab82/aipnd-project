import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms, models
from torch import nn, optim, tensor
import torch.nn.functional as F
from collections import OrderedDict
from PIL import Image
from torch.autograd import Variable

import helperr
import json
import os
import copy
import time

import argparse



ap = argparse.ArgumentParser(description='Train.py')


ap.add_argument('data_dir', action="store", default="./flowers/")
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=3)
ap.add_argument('--model', dest="model", action="store", default="densenet121", type = str)
ap.add_argument('--hidden_layer1', type=int, dest="hidden_layer1", action="store", default=4096)



pa = ap.parse_args()
root = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
model = pa.model
dropout = pa.dropout
hidden_layer1 = pa.hidden_layer1
device = pa.gpu
epochs = pa.epochs

def main():
    
    trainloader, validloader, testloader = helperr.load_data(root)
    model, optimizer, criterion = helperr.build_model(hidden_layer1, class_to_idx)
    helperr.train(model, epochs, lr, criterion, optimizer, trainloader, validloader)
    helperr.save_checkpoint(model,path,structure,hidden_layers,dropout,lr)
    print("Done Training!")


if __name__== "__main__":
    main()
