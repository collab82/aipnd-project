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

def nn_setup(structure='densenet121',dropout=0.5, hidden_layer1 = 512,lr = 0.001):
    
    
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

        return model , optimizer ,criterion

def train(model, epochs = 3, learning_rate, criterion, optimizer, trainloader, validloader):
    
    model.train()
    print_every = 40
    steps = 0
    use_gpu = False
    
    if torch.cuda.is_available():
        use_gpu = True
        model.cuda()
    else:
        model.cpu()

    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in iter(trainloader):
            steps += 1

            if use_gpu:
                inputs = Variable(inputs.float().cuda())
                labels = Variable(labels.long().cuda()) 
            else:
                inputs = Variable(inputs)
                labels = Variable(labels) 

            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]

            if steps % print_every == 0:
                validation_loss, accuracy = validate(model, criterion, validloader)

                
                print("-------------- Finished training -----------------------")

                print("Epoch: {}/{} ".format(epoch+1, epochs),
                        "Training Loss: {:.3f} ".format(running_loss/print_every),
                        "Validation Loss: {:.3f} ".format(validation_loss),
                        "Validation Accuracy: {:.3f}".format(accuracy))
                print("That's a lot of steps")
                
                
def save_checkpoint(path='checkpoint.pth',structure ='densenet121', hidden_layer1=512,dropout=0.5,lr=0.001,epochs=12):
    model.class_to_idx = image_datasets['train_data'].class_to_idx
    model.cpu
        
    torch.save({'structure': 'densenet121',
              'input_size': (3, 224, 224),
              'output_size': 102,
              'hidden_layer1': 512,
              'model': model,
              'state_dict': model.state_dict(),
              'optimizer': optimizer,
              'epoch': epochs,
              'lr' : learning_rate,
              'class_to_idx': model.class_to_idx},
               'checkpoint.pth')

def load_checkpoint(path='checkpoint.pth):
    checkpoint = torch.load(path)
    structure = checkpoint['structure']
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    hidden_layer1 = checkpoint['hidden_layer1']
    model,_,_ = nn_setup(structure , 0.5,hidden_layer1)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])


def process_image(image_path):
    for i in image:
        path = str(i)
    img = Image.open(i)

    image = image.resize((256, 256))
    image = image.crop((16, 16, 240, 240))
    image = np.array(image)
    image = image/255.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    image = np.transpose(image, (2, 0, 1))
    
    return image.astype(np.float32)
                    
def predict(image_path, model, topk=5):

    img = Image.open(image_path)
    img = process_image(img)
    
    img = np.expand_dims(img, 0)
    
    img = torch.from_numpy(img)
                    
    model.eval()
    inputs = Variable(img).to(device)
    logits = model.forward(inputs)
    
    ps = F.softmax(logits,dim=1)
    topk = ps.cpu().topk(topk)
    
    return (e.data.numpy().squeeze().tolist() for e in topk)

