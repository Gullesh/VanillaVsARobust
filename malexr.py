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
from alexp import *
import alx
import alexp
from torchray.attribution.extremal_perturbation import extremal_perturbation, contrastive_reward, simple_reward
from torchray.attribution.grad_cam import grad_cam
from captum.attr import IntegratedGradients
from torchray.attribution.rise import rise
from lime import lime_image


model_options = ['resnet50', 'resnet50r', 'googlenet', 'googlenetr','vgg16', 'alexnet', 'alexnetr']
method_options = ['gradient', 'mp','ep', 'gradcam', 'rise','ig', 'rise', 'lime','shap']
data_options = ['rnddata', 'rsnt', 'ggl', 'alx']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--model', '-a', default='resnet50',  choices=model_options)
parser.add_argument('--method', '-m', default='gradient',  choices=method_options)
parser.add_argument('--data', '-d', default='rnddata',  choices=data_options)

parser.add_argument('--mpiteration', type=int, default=300, help='number of iteration for mp method')
parser.add_argument('--tol', type=int, default=1, help='tolerance for  pointing game')

args = parser.parse_args()
# Loading model

if args.model == 'alexnetr':
    model = alx.alexr()
    target_layer = model.model.features[11]

else:
    print('Error: please choose a valid model')


if args.data == 'rnddata':
    dta = '/home/mallet/Desktop/Dataa/rnddata/valid'
    bdta = '/home/mallet/Desktop/Dataa/rnddata/val/'
elif args.data == 'alx':
    dta = '/home/mallet/Desktop/Dataa/alx/valid'
    bdta = '/home/mallet/Desktop/Dataa/alx/val/'
else:
    print('Error: please choose a valid data')

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
    val_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])    
    # 2000 images randomly taken from Imagenet(ILSVRC2012) validation set
    vall = torchray.benchmark.datasets.ImageFolder(dta,transform = val_transforms)
    # number of images we need to calculate things for
    nimg = vall.selection
    for i in nimg:
        img, labele = vall[i]
        bbname = vall.get_image_url(i).split("/")[-1].split(".")[0]             # Extracting image url to retrieve its BB
        imwidth, imheight, xmin, ymin, xmax, ymax = miscel.bbinfo(bbname,bdta)
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
    val_transforms = transforms.Compose([transforms.Resize((224,224))])    
    # 2000 images randomly taken from Imagenet(ILSVRC2012) validation set
    vall = torchray.benchmark.datasets.ImageFolder(dta,transform = val_transforms)
    # number of images we need to calculate things for
    nimg = vall.selection

    for i in nimg:
        
        img, labele = vall[i]     
        bbname = vall.get_image_url(i).split("/")[-1].split(".")[0]             # Extracting image url to retrieve its BB
        imwidth, imheight, xmin, ymin, xmax, ymax = miscel.bbinfo(bbname,bdta)
        xminn, yminn, xmaxn, ymaxn = miscel.newloc(imwidth, imheight, xmin, ymin, xmax, ymax)
        ti, saliency = alexp.mp(model, img, labele, args.mpiteration)
        ttotal = ttotal + ti
        z = saliency, labele, bbname
        lslb.append(z)
        l.append(ti)
        xloc,yloc = miscel.findloc(saliency)
        Y = miscel.gtbb(xminn, yminn, xmaxn, ymaxn)
        pg = test.evaluate(Y, (yloc,xloc))
        if pg==1:
            pghits+=1
        elif pg==-1:
            pgmiss+=1

elif args.method == 'ep':
    # Transforms needs to be applied to our data set
    val_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])  
    # 2000 images randomly taken from Imagenet(ILSVRC2012) validation set
    vall = torchray.benchmark.datasets.ImageFolder(dta,transform = val_transforms)


    # number of images we need to calculate things for
    nimg = vall.selection
        
    for i in nimg:
        img, labele = vall[i]
        bbname = vall.get_image_url(i).split("/")[-1].split(".")[0]             # Extracting image url to retrieve its BB
        imwidth, imheight, xmin, ymin, xmax, ymax = miscel.bbinfo(bbname, bdta)
        xminn, yminn, xmaxn, ymaxn = miscel.newloc(imwidth, imheight, xmin, ymin, xmax, ymax)
   
        x = img.unsqueeze(0)
        x = x.cuda()
        start = time.time()
        saliency, _ = extremal_perturbation(model, x, labele, reward_func=simple_reward, debug=False, areas=[0.025, 0.05, 0.1, 0.2],smooth=0.09,perturbation='blur')
        end = time.time()
        tm = end - start
        mask = saliency.sum(dim=0, keepdim=True)
        ttotal = ttotal + tm
        z = mask, labele, bbname
        lslb.append(z)
        l.append(tm)
        xloc,yloc = miscel.findloc(mask)
        Y = miscel.gtbb(xminn, yminn, xmaxn, ymaxn)
        pg = test.evaluate(Y, (yloc,xloc))
        if pg==1:
            pghits+=1
        elif pg==-1:
            pgmiss+=1



elif args.method == 'gradcam':
    # Transforms needs to be applied to our data set
    val_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])  
    # 2000 images randomly taken from Imagenet(ILSVRC2012) validation set
    vall = torchray.benchmark.datasets.ImageFolder(dta,transform = val_transforms)
    aaa = 0  

    # Make it to be able to get gradient
    for p in model.parameters():
        p.requires_grad = True

    # Initialization of GradCam method
    cam = GradCAM(model=model, target_layer=target_layer, use_cuda=True)

    # number of images we need to calculate things for
    nimg = vall.selection
    for i in nimg:
        img, labele = vall[i]
        

        bbname = vall.get_image_url(i).split("/")[-1].split(".")[0]             # Extracting image url to retrieve its BB

        imwidth, imheight, xmin, ymin, xmax, ymax = miscel.bbinfo(bbname, bdta)
        xminn, yminn, xmaxn, ymaxn = miscel.newloc(imwidth, imheight, xmin, ymin, xmax, ymax)

        x = img.unsqueeze(0)
        x = x.cuda()
        start = time.time()
        saliency = cam(input_tensor=x, target_category=labele)
        end = time.time()
        tm = end - start
        ttotal = ttotal + tm
        z = saliency, labele, bbname
        lslb.append(z)
        l.append(tm)
        xloc,yloc = miscel.gradfindloc(saliency)
        Y = miscel.gtbb(xminn, yminn, xmaxn, ymaxn)
        pg = test.evaluate(Y, (yloc,xloc))
        if pg==1:
            pghits+=1
        elif pg==-1:
            pgmiss+=1
    val_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])    

elif args.method == 'ig':
    # Transforms needs to be applied to our data set
    val_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])    
   
    # 2000 images randomly taken from Imagenet(ILSVRC2012) validation set
    vall = torchray.benchmark.datasets.ImageFolder(dta,transform = val_transforms)

    # Method
    ig = IntegratedGradients(model)

    # number of images we need to calculate things for
    nimg = vall.selection
    for i in nimg:
        img, labele = vall[i]
        bbname = vall.get_image_url(i).split("/")[-1].split(".")[0]             # Extracting image url to retrieve its BB
        imwidth, imheight, xmin, ymin, xmax, ymax = miscel.bbinfo(bbname, bdta)
        xminn, yminn, xmaxn, ymaxn = miscel.newloc(imwidth, imheight, xmin, ymin, xmax, ymax)
        x = img.unsqueeze(0)
        x = x.cuda()
        start = time.time()
        attributions = ig.attribute(x, target=labele)
        saliency = torch.mean(attributions, 1,keepdim=True)
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

elif args.method == 'rise':
    # Transforms needs to be applied to our data set
    val_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])    
    # 2000 images randomly taken from Imagenet(ILSVRC2012) validation set
    vall = torchray.benchmark.datasets.ImageFolder(dta,transform = val_transforms)


    # number of images we need to calculate things for
    nimg = vall.selection
    for i in nimg:
        img, labele = vall[i]
        bbname = vall.get_image_url(i).split("/")[-1].split(".")[0]             # Extracting image url to retrieve its BB
        imwidth, imheight, xmin, ymin, xmax, ymax = miscel.bbinfo(bbname, bdta)
        xminn, yminn, xmaxn, ymaxn = miscel.newloc(imwidth, imheight, xmin, ymin, xmax, ymax)
        x = img.unsqueeze(0)
        x = x.cuda()
        start = time.time()
        saliency = rise(model, x)
        saliency = saliency[:, labele].unsqueeze(0)
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
elif args.method == 'lime':

    def get_preprocess_transform():    
        transf = transforms.Compose([
            transforms.ToTensor()
        ])    

        return transf

    preprocess_transform = get_preprocess_transform()

    def batch_predict(images):
        batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        batch = batch.to(device)
        
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()


    # Transforms needs to be applied to our data set
    val_transforms = transforms.Compose([transforms.Resize((224,224))])    
    # 2000 images randomly taken from Imagenet(ILSVRC2012) validation set
    vall = torchray.benchmark.datasets.ImageFolder(dta,transform = val_transforms)

    # Method
    explainer = lime_image.LimeImageExplainer()

    # number of images we need to calculate things for
    nimg = vall.selection
    for i in nimg:
        img, labele = vall[i]
        bbname = vall.get_image_url(i).split("/")[-1].split(".")[0]             # Extracting image url to retrieve its BB
        imwidth, imheight, xmin, ymin, xmax, ymax = miscel.bbinfo(bbname, bdta)
        xminn, yminn, xmaxn, ymaxn = miscel.newloc(imwidth, imheight, xmin, ymin, xmax, ymax)

        start = time.time()
        explanation = explainer.explain_instance(np.array(img), 
                                                batch_predict, # classification function
                                                labels = labele,
                                                top_labels=None,  
                                                num_samples=1000)
        _, mask = explanation.get_image_and_mask(label=labele, positive_only=False, num_features=5, negative_only=False, hide_rest=False)

        end = time.time()
        tm = end - start
        ttotal = ttotal + tm
        z = mask, labele, bbname
        lslb.append(z)
        l.append(tm)
        #saliency = miscel.normlze(saliency)
        xloc,yloc = miscel.camfindloc(mask)
        Y = miscel.gtbb(xminn, yminn, xmaxn, ymaxn)
        pg = test.evaluate(Y, (yloc,xloc))
        if pg==1:
            pghits+=1
        elif pg==-1:
            pgmiss+=1



torch.save(lslb, '/home/mallet/Desktop/Dataa/salmaps/'+args.method +args.model+args.data+'sal.pt')
l = np.asarray(l)     
savetxt('/home/mallet/Desktop/Dataa/Runtimes/time'+args.method+args.model+args.data+'.csv', l, delimiter=',')
print('Pointing game accuracy: ',pghits/(pghits+pgmiss))
print('Total time: ',ttotal)
print('hits and misses: ', pghits, pgmiss)
