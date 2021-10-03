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
import argparse
from exp import *
import exp
from torchray.attribution.extremal_perturbation import extremal_perturbation, contrastive_reward, simple_reward
from torchray.attribution.grad_cam import grad_cam

model_options = ['resnet50', 'resnet50r', 'googlenet', 'googlenetr','vgg16']
method_options = ['gradient', 'mp','ep', 'gradcam', 'rise', 'scorecam']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--model', '-a', default='resnet50',  choices=model_options)
parser.add_argument('--method', '-m', default='gradient',  choices=method_options)
parser.add_argument('--mpiteration', type=int, default=500, help='number of iteration for mp method')
parser.add_argument('--tol', type=int, default=15, help='tolerance for  pointing game')

args = parser.parse_args()
# Loading model
if args.model == 'resnet50':
    model = lm.loadResnet()
elif args.model == 'vgg16':
    model = lm.loadvgg()
elif args.model == 'resnet50r':
    model = lm.loadResnetR()
elif args.model == 'googlenet':
    model = lm.loadgoogle()
elif args.model == 'googlenetr':
    model = lm.loadgoogleR()
else:
    print('Error: please choose a valid model')


tol = args.tol
# Initializing number of hits in pointing game and total time
ttotal = 0
pghits = 0
pgmiss = 0
test = torchray.benchmark.pointing_game.PointingGame(1000, tolerance=tol)

# Initializing list for runtime
l = []

# initializing list for saliency maps, labels, bbname
lslb = []


if args.method == 'gradient':
    # Transforms needs to be applied to our data set
    val_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])    
    # 2000 images randomly taken from Imagenet(ILSVRC2012) validation set
    vall = torchray.benchmark.datasets.ImageFolder('/home/mallet/Desktop/VanillaVsARobust/validationSample',transform = val_transforms)
    # number of images we need to calculate things for
    nimg = vall.selection
    aaa = 0
    for i in nimg:
        img, labele = vall[i]
        bbname = vall.get_image_url(i).split("/")[-1].split(".")[0]             # Extracting image url to retrieve its BB
        imwidth, imheight, xmin, ymin, xmax, ymax = miscel.bbinfo(bbname)
        xminn, yminn, xmaxn, ymaxn = miscel.newloc(imwidth, imheight, xmin, ymin, xmax, ymax)
        x = img.unsqueeze(0)
        x = x.cuda()
        start = time.time()
        saliency = gradient(model, x, labele)
        end = time.time()
        tm = end - start
        ttotal = ttotal + tm
        z = saliency, labele, bbname
        lslb.append(z)
        l.append(tm)
        #saliency = miscel.normlze(saliency)
        xloc,yloc = miscel.findloc(saliency)
        Y = miscel.gtbb(xminn, yminn, xmaxn, ymaxn)
        pg = test.evaluate(Y, (yloc,xloc))
        if pg==1:
            pghits+=1
        elif pg==-1:
            pgmiss+=1


elif args.method == 'mp':

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

    for i in nimg:
        img, labele = vall[i]     
        I = np.asarray(img)                          
        bbname = vall.get_image_url(i).split("/")[-1].split(".")[0]             # Extracting image url to retrieve its BB
        imwidth, imheight, xmin, ymin, xmax, ymax = miscel.bbinfo(bbname)   
        ti, saliency = mp(model, img, labele, args.mpiteration)
        exp.save(saliency, I, args.method, args.model, bbname)
        ttotal = ttotal + ti
        z = saliency, labele, bbname
        lslb.append(z)
        l.append(ti)
        xloc,yloc = miscel.findloc(saliency, imwidth, imheight)
        Y = miscel.gtbb(imwidth, imheight, xmin, ymin, xmax, ymax)
        test = torchray.benchmark.pointing_game.PointingGame(1000, tolerance=15)
        pg = test.evaluate(Y, (yloc,xloc))
        if pg==1:
            pgtot+=1

elif args.method == 'ep':
    # Transforms needs to be applied to our data set
    val_transforms = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])    
    # 2000 images randomly taken from Imagenet(ILSVRC2012) validation set
    vall = torchray.benchmark.datasets.ImageFolder('/home/mallet/Desktop/VanillaVsARobust/validationSample',transform = val_transforms)

    # Initializing number of hits in pointing game and total time
    ttotal = 0
    pgtot = 0

    # Initializing list for runtime
    l = []

    # number of images we need to calculate things for
    nimg = vall.selection
        
    for i in nimg:
        img, labele = vall[i]
        I = np.asarray(img)                                                 
        bbname = vall.get_image_url(i).split("/")[-1].split(".")[0]             # Extracting image url to retrieve its BB
        imwidth, imheight, xmin, ymin, xmax, ymax = miscel.bbinfo(bbname)   
        x = img.unsqueeze(0)
        x = x.cuda()
        start = time.time()
        saliency,_ = extremal_perturbation( model, x, labele, reward_func=simple_reward, debug=False, areas=[0.12])
        end = time.time()
        I = np.transpose(I, (1, 2, 0))
        exp.save(saliency, I, args.method, args.model, bbname)
        tm = end - start
        ttotal = ttotal + tm
        z = saliency, labele, bbname
        lslb.append(z)
        l.append(tm)
        xloc,yloc = miscel.findloc(saliency)
        Y = miscel.gtbb(imwidth, imheight, xmin, ymin, xmax, ymax)
        test = torchray.benchmark.pointing_game.PointingGame(1000, tolerance=15)
        pg = test.evaluate(Y, (yloc,xloc))
        if pg==1:
            pgtot+=1

torch.save(lslb, '/home/mallet/Desktop/Dataa/salmaps/'+args.method +args.model+'sal.pt')
l = np.asarray(l)     
savetxt('/home/mallet/Desktop/Dataa/Runtimes/time'+args.method+args.model+'.csv', l, delimiter=',')
print('Pointing game accuracy: ',pghits/(pghits+pgmiss))
print('Total time: ',ttotal)
print('hits and misses: ', pghits, pgmiss)
