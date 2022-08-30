import sys
import os
import sys
sys.path.append('/home/mallet/Desktop/shapebias')
os.chdir('/home/mallet/Desktop/shapebias')
from settings import settings
from shapebias.robustness.model_utils import make_and_restore_model
from PIL import Image
from shapebias.robustness.datasets import ImageNet
import torch.nn as nn
from torchvision import models
from robustness import model_utils, datasets, train, defaults
#from robustness.datasets import ImageNet
import torch

use_cuda = torch.cuda.is_available()

def alexr():
    dataset = ImageNet('/home/mallet/Downloads/imgnet')
    model, _ = make_and_restore_model(arch=settings.MODEL[:-2], dataset=dataset,parallel=settings.MODEL_PARALLEL, resume_path=settings.MODEL_PATH)
    
    #model = nn.Sequential(model, nn.Softmax(dim=1))

    return model

def loadmobr():
        ds = ImageNet('/media/mallet/Haniye/data/')
        densenet , _ = model_utils.make_and_restore_model(arch=models.mobilenet_v2(), dataset=ds,  resume_path='/media/mallet/Haniye/mobilenet_l2_eps3.ckpt', add_custom_forward=True,parallel=False)
        model = densenet.eval()
        model.cuda()
        for p in model.parameters():
                p.requires_grad = False
        
        return model

def loadmob():
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    if use_cuda:
        model.cuda()

    for p in model.parameters():
        p.requires_grad = False

    return model

def loaddensr():
        ds = ImageNet('/media/mallet/Haniye/data/')
        densenet , _ = model_utils.make_and_restore_model(arch=models.densenet161(), dataset=ds,  resume_path='/media/mallet/Haniye/densenet_l2_eps3.ckpt', add_custom_forward=True,parallel=False)
        model = densenet.eval()
        model.cuda()
        for p in model.parameters():
                p.requires_grad = False
        
        return model

def loaddens():
    model = models.densenet161(pretrained=True)
    model.eval()
    if use_cuda:
        model.cuda()

    for p in model.parameters():
        p.requires_grad = False

    return model