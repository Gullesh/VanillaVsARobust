import numpy as np
import torchvision.transforms as transforms
from torchray.attribution.gradient import gradient
import torchray.benchmark
import xml.etree.ElementTree as ET
import miscel
import torchray.benchmark.pointing_game
import miscel
import time
from numpy import savetxt
import argparse
import gc
import torch

data_options = ['rnddata', 'rsnt', 'ggl', 'alx']

parser = argparse.ArgumentParser(description='CNN')

parser.add_argument('--data', '-d', default='rnddata',  choices=data_options)

args = parser.parse_args()


if args.data == 'rnddata':
    dta = '/home/mallet/Desktop/Dataa/rnddata/valid'
    bdta = '/home/mallet/Desktop/Dataa/rnddata/val/'
elif args.data == 'rsnt':
    dta = '/home/mallet/Desktop/Dataa/rsnt/valid'
    bdta = '/home/mallet/Desktop/Dataa/rsnt/val/'
elif args.data == 'ggl':
    dta = '/home/mallet/Desktop/Dataa/ggl/valid'
    bdta = '/home/mallet/Desktop/Dataa/ggl/val/'
elif args.data == 'alx':
    dta = '/home/mallet/Desktop/Dataa/alx/valid'
    bdta = '/home/mallet/Desktop/Dataa/alx/val/'
else:
    print('Error: please choose a valid data')
 

data = torch.load('/home/mallet/Desktop/Dataa/salmaps/gradcamresnet50rsntsal.pt')
tol = [1, 5, 10, 15, 20, 25]

for i in tol:
    pghits = 0
    pgmiss = 0
    test = torchray.benchmark.pointing_game.PointingGame(1000, tolerance=i)

    for j in range(len(data)):
        
        sal, _ , bbname = data[j]
        if torch.is_tensor(sal):
            saliency = sal
        else:
            saliency = torch.from_numpy(sal)

        imwidth, imheight, xmin, ymin, xmax, ymax = miscel.bbinfo(bbname, bdta)
        xminn, yminn, xmaxn, ymaxn = miscel.newloc(imwidth, imheight, xmin, ymin, xmax, ymax)

        xloc,yloc = miscel.findloc(saliency)
        Y = miscel.gtbb(xminn, yminn, xmaxn, ymaxn)
        pg = test.evaluate(Y, (yloc,xloc))
        if pg==1:
            pghits+=1
        elif pg==-1:
            pgmiss+=1
    print('Pointing game accuracy for tol '+ str(i)+':' ,pghits/(pghits+pgmiss))

    