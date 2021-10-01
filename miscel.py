# Misc functions
import xml.etree.ElementTree as ET
import torch
import numpy as np
import cv2

# Obtaining groundtruth bounding box info
def bbinfo(bbname):
    tree = ET.parse('/home/mallet/Desktop/VanillaVsARobust/bb/'+ bbname + '.xml')
    root = tree.getroot()
    imwidth = int(root[3][0].text)
    imheight = int(root[3][1].text)
    xmin = int(root[5][4][0].text)
    ymin = int(root[5][4][1].text)
    xmax = int(root[5][4][2].text)
    ymax = int(root[5][4][3].text)

    return imwidth, imheight, xmin, ymin, xmax, ymax

# Finding location of highest intensity pixel

def findloc(saliency, imwidth, imheight):
    saliency = torch.squeeze(saliency)
    saliency = saliency.cpu().detach().numpy()
    salresized = cv2.resize(saliency, (imwidth, imheight))
    xloc,yloc = np.unravel_index(salresized.argmax(), salresized.shape)

    return xloc, yloc

# Constructing groundtruth bounding box for torchray's pointing game 
def gtbb(imwidth, imheight, xmin, ymin, xmax, ymax):
    a = np.zeros((imheight, imwidth))
    a[ymin:ymax+1, xmin:xmax+1] = 1
    Y = torch.zeros((imheight, imwidth), dtype=torch.bool)
    Y[ymin:ymax+1, xmin:xmax+1]=True
    return Y, a

def sares(saliency, imwidth, imheight):
    saliency = torch.squeeze(saliency)
    saliency = saliency.cpu().detach().numpy()
    salresized = cv2.resize(saliency, (imwidth, imheight))
    return salresized