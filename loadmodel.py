# Loading models

import torch 
from torchvision import models
import sys
import os
sys.path.append('/home/mallet/Desktop/sam')
os.chdir('/home/mallet/Desktop/sam')
from sam.utils import load_madry_model
import torch.nn as nn

use_cuda = torch.cuda.is_available()

# load models


def loadgoogle():
    model = models.googlenet(pretrained=True)
    #model = nn.Sequential(model, nn.Softmax(dim=1))
    model.eval()
    if use_cuda:
        model.cuda()

    for p in model.parameters():
        p.requires_grad = False

        
    return model
def loadvgg():
    model = models.vgg16(pretrained=True)
    model.eval()
    if use_cuda:
        model.cuda()

    for p in model.parameters():
        p.requires_grad = False

    return model


def loadResnet():
    model = models.resnet50(pretrained=True)
    #model = nn.Sequential(model, nn.Softmax(dim=1))
    model.eval()
    if use_cuda:
        model.cuda()

    for p in model.parameters():
        p.requires_grad = False

    return model

def loadAlexnet():
    model = models.alexnet(pretrained=True)
    #model = nn.Sequential(model, nn.Softmax(dim=1))
    model.eval()
    if use_cuda:
        model.cuda()

    for p in model.parameters():
        p.requires_grad = False

    return model

def loadResnetR():
    model = load_madry_model(arch='madry', if_pre=1, my_attacker=True)
    model.model.eval()
    if use_cuda:
        model.cuda()

    for p in model.parameters():
        p.requires_grad = False

    return model

def loadgoogleR():
    model = load_madry_model(arch='madry_googlenet', if_pre=1, my_attacker=True)
    model.eval()
    if use_cuda:
        model.cuda()

    for p in model.parameters():
        p.requires_grad = False
    return model