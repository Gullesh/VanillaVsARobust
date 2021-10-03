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


# Modifying the location coordinates black box since we resized the image!

def newloc(imwidth, imheight, xmin, ymin, xmax, ymax):

    x_scale = 224 / imwidth
    y_scale = 224 / imheight
    xminn = int(np.round(xmin * x_scale))
    xmaxn = int(np.round(xmax * x_scale))
    yminn = int(np.round(ymin * y_scale))
    ymaxn = int(np.round(ymax * y_scale))

    return xminn, yminn, xmaxn, ymaxn

# Finding location of highest intensity pixel

def findloc(saliency):
    saliency = torch.squeeze(saliency)
    saliency = saliency.cpu().detach().numpy()
    if saliency.shape != (224, 224):
        saliency = cv2.resize(saliency, (224, 224))
    xloc,yloc = np.unravel_index(saliency.argmax(), (224,224))

    return xloc, yloc

# Constructing groundtruth bounding box for torchray's pointing game 
def gtbb(xmin, ymin, xmax, ymax):
   
    Y = torch.zeros((224, 224), dtype=torch.bool)
    Y[ymin:ymax+1, xmin:xmax+1]=True
    return Y

def sares(saliency, imwidth, imheight):
    saliency = torch.squeeze(saliency)
    saliency = saliency.cpu().detach().numpy()
    salresized = cv2.resize(saliency, (imwidth, imheight))
    return salresized
# Normalize between 0 and 1
def normlze(saliency):
    nmlized = (saliency - saliency.min()) /saliency.max()
    return nmlized

