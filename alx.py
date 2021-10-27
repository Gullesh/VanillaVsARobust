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

def alexr():
    dataset = ImageNet('/home/mallet/Downloads/imgnet')
    model, _ = make_and_restore_model(arch=settings.MODEL[:-2], dataset=dataset,parallel=settings.MODEL_PARALLEL, resume_path=settings.MODEL_PATH)
    
    #model = nn.Sequential(model, nn.Softmax(dim=1))

    return model