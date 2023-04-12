import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
import json

transformm=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([64, 128]),
    transforms.Normalize((0.5),(0.5)),

    ])
model=torch.load('./model/MSRF_firemaker_IAM_model_vertical_aug_16-model_epoch_3.pth')
#for n,c in model.named_children():
    #print(" Layer Name: ",n,)

#print(model)
# we will save the conv layer weights in this list
model_weights =[]
#we will save the 49 conv layers in this list
conv_layers = []
# get all the model children as list
model_children = list(model.children())
#model_child=list(model_children.children())
#print("model children ", model_children)
#counter to keep count of the conv layers
counter = 0

no_of_layers = 0
conv_layers = []


for child in model_children:
    if type(child) == nn.Conv2d:
        no_of_layers += 1
        conv_layers.append(child)
    elif type(child) == nn.Sequential:
        for layer in child.children():
            if type(layer) == nn.Conv2d:
                no_of_layers += 1
                conv_layers.append(layer)
print(no_of_layers)