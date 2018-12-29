import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
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


ap = argparse.ArgumentParser(description='Predict.py')

ap.add_argument('input', default='./flowers/test/1/image_06752.jpg', nargs='?', action="store", type = str)
ap.add_argument('--dir', action="store",dest="data_dir", default="./flowers/")
ap.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--gpu', default="gpu", action="store", dest="gpu")

pa = ap.parse_args()
imp_path = pa.input
number_of_outputs = pa.top_k
device = pa.gpu

path = pa.checkpoint

pa = ap.parse_args()

def main():
    model=helperr.load_checkpoint(path)
    with open('cat_to_name.json', 'r') as json_file:
        cat_to_name = json.load(json_file)
    probabilities = helperr.predict(img_path, model, number_of_outputs, device)
    labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
    probability = np.array(probabilities[0][0])
    i=0
    while i < number_of_outputs:
        print("{} with a probability of {}".format(labels[i], probability[i]))
        i += 1
    print("Done Predicting!")

    
if __name__== "__main__":
    main()
