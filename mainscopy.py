import numpy as np
import torchvision.transforms as transforms
from torchray.attribution.gradient import gradient
import torchray.benchmark
import xml.etree.ElementTree as ET
import miscel
import torchray.benchmark.pointing_game
import miscel
import loadmodel as lm
import time
from numpy import savetxt
from exp import *
import exp
# Transforms needs to be applied to our data set
val_transforms = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224)])    
# 2000 images randomly taken from Imagenet(ILSVRC2012) validation set
vall = torchray.benchmark.datasets.ImageFolder('/home/mallet/Desktop/VanillaVsARobust/validationSample',transform = val_transforms)

# Initializing number of hits in pointing game and total time
ttotal = 0
pgtot = 0

# Initializing list for runtime
l = []

# number of images we need to calculate things for
nimg = vall.selection

# Loading model
model = lm.loadResnet()

for i in nimg:
    img, labele = vall[i]     
    I = np.asarray(img)                          
    bbname = vall.get_image_url(i).split("/")[-1].split(".")[0]             # Extracting image url to retrieve its BB
    imwidth, imheight, xmin, ymin, xmax, ymax = miscel.bbinfo(bbname)   
    #x = img.unsqueeze(0)
    #x = x.cuda()
    #start = time.time()
    #saliency = gradient(model, x, labele)
    #print(img.shape)
    ti, saliency = mp(model, img, labele, 500)
    exp.save(saliency, I, bbname)
    #end = time.time()
    #tm = end - start
    ttotal = ttotal + ti
    l.append(ti)
    xloc,yloc = miscel.findloc(saliency, imwidth, imheight)
    Y = miscel.gtbb(imwidth, imheight, xmin, ymin, xmax, ymax)
    test = torchray.benchmark.pointing_game.PointingGame(1000, tolerance=15)
    pg = test.evaluate(Y, (yloc,xloc))
    if pg==1:
        pgtot+=1

l = np.asarray(l)     
savetxt('/home/mallet/Desktop/VanillaVsARobust/Runtimes/timempres.csv', l, delimiter=',')
print('Pointing game accuracy: ',pgtot/len(vall.selection))
print('Total time: ',ttotal)

