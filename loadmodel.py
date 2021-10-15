# Loading models

import torch 
from torchvision import models
import sys
import os
sys.path.append('/home/mallet/Desktop/sam')
os.chdir('/home/mallet/Desktop/sam')
from sam.utils import load_madry_model

use_cuda = torch.cuda.is_available()

# load models

def loadgoogle():
    model = models.googlenet(pretrained=True)
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
    model.eval()
    if use_cuda:
        model.cuda()

    for p in model.parameters():
        p.requires_grad = False

    return model

def loadAlexnet():
    model = models.alexnet(pretrained=True)
    model.eval()
    if use_cuda:
        model.cuda()

    for p in model.parameters():
        p.requires_grad = False

    return model

def loadResnetR():
    model = load_madry_model(arch='madry', my_attacker=True)
    return model

def loadgoogleR():
    model = load_madry_model(arch='madry_googlenet', my_attacker=True)
    return model