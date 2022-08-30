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
from pgexp import *
import pgexp
from pytorch_grad_cam import GradCAM
import torch.nn.functional as F
from torchray.attribution.extremal_perturbation import extremal_perturbation, simple_reward
from captum.attr import IntegratedGradients
from lime import lime_image
import shap
from RISE.explanations import RISE
from RISE.utils import *
from lime.wrappers.scikit_image import SegmentationAlgorithm
from scipy.stats.stats import pearsonr   
import torch.nn as nn
import load_model as llm
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


model_options = ['resnet50', 'pg1', 'pg5']
method_options = ['gradient', 'mp','ep', 'gradcam', 'rise', 'cam','ig', 'rise', 'lime','shap','mpp','epp','lim','riser']
data_options = ['rnddata', 'rsnt','hyp']

parser = argparse.ArgumentParser(description='CNN')

parser.add_argument('--model', '-a', default='resnet50',  choices=model_options)
parser.add_argument('--method', '-m', default='gradient',  choices=method_options)
parser.add_argument('--data', '-d', default='rnddata',  choices=data_options)
parser.add_argument('--mpiteration', type=int, default=300, help='number of iteration for mp method')
parser.add_argument('--tol', type=int, default=1, help='tolerance for  pointing game')

args = parser.parse_args()
# Loading model
if args.model == 'resnet50':
    model = lm.loadResnet()
    #target_layer = model.layer4
    #modulll = model._modules.get('layer4')
    #params = list(model.parameters())
    #weight_softmax = np.squeeze(params[-2].cpu().data.numpy())
elif args.model == 'pg5':
    model = llm.load_adv_prop_model('/home/mallet/Desktop/VanillaVsARobust/pgd_5.pth.tar')
    model.cuda()
    model.eval()
    target_layers = model.module.layer4
    modulll = model._modules.get('module')._modules.get('layer4')
    params = list(model.module.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())
elif args.model == 'pg1':
    model = llm.load_adv_prop_model('/home/mallet/Desktop/VanillaVsARobust/pgd_1.pth.tar')
    model.cuda()
    model.eval()
    target_layers = model.module.layer4
    modulll = model._modules.get('module')._modules.get('layer4')
    params = list(model.module.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())


else:
    print('Error: please choose a valid model')

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
elif args.data == 'subalx':
    dta = '/home/mallet/Desktop/Dataa/subalx/valid'
    bdta = '/home/mallet/Desktop/Dataa/subalx/val/'
elif args.data == 'subggl':
    dta = '/home/mallet/Desktop/Dataa/subggl/valid'
    bdta = '/home/mallet/Desktop/Dataa/subggl/val/'
elif args.data == 'hyp':
    dta = '/home/mallet/Desktop/Dataa/hyp/valid'
    bdta = '/home/mallet/Desktop/Dataa/hyp/val/'

elif args.data == 'subrsnt':
    dta = '/home/mallet/Desktop/Dataa/subrsnt/valid'
    bdta = '/home/mallet/Desktop/Dataa/subrsnt/val/'

else:
    print('Error: please choose a valid data')
 
tol = args.tol
# Initializing number of hits in pointing game and total time
ttotal = 0
pghits = 0
pgmiss = 0
test = torchray.benchmark.pointing_game.PointingGame(1000, tolerance=tol)

#lists for confidences and metrics
confid = []
metri1 = []
metri2 = []

# Initializing list for runtime
l = []

# initializing list for saliency maps, labels, bbname
lslb = []


if args.method == 'gradient':
    # Transforms needs to be applied to our data set
    val_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])    
    # 2000 images randomly taken from Imagenet(ILSVRC2012) validation set
    vall = torchray.benchmark.datasets.ImageFolder(dta,transform = val_transforms)
    # number of images we need to calculate things for
    nimg = vall.selection
    for i in nimg:
        img, labele = vall[i]
        bbname = vall.get_image_url(i).split("/")[-1].split(".")[0]             # Extracting image url to retrieve its BB
        imwidth, imheight, xmin, ymin, xmax, ymax = miscel.bbinfo(bbname, bdta)
        xminn, yminn, xmaxn, ymaxn = miscel.newloc(imwidth, imheight, xmin, ymin, xmax, ymax)
        x = img.unsqueeze(0).cuda()
        start = time.time()
        saliency = gradient(model, x, labele)
        end = time.time()
        tm = end - start
        ttotal = ttotal + tm
        l.append(tm)
        z = saliency, labele, bbname
        lslb.append(z)
        sal = torch.squeeze(saliency).cpu().detach().numpy()
        Y = miscel.gtbb(xminn, yminn, xmaxn, ymaxn)
        maxmat = sal == sal.max()
        hit = (torch.from_numpy(maxmat) & Y).view(-1).any()
        if hit:
            pghits+=1
        else:
            pgmiss+=1


elif args.method == 'mp':
    model = model.eval()
    # Transforms needs to be applied to our data set
    val_transforms = transforms.Compose([transforms.Resize((224,224))])    
    # 2000 images randomly taken from Imagenet(ILSVRC2012) validation set
    vall = torchray.benchmark.datasets.ImageFolder(dta,transform = val_transforms)
    # number of images we need to calculate things for
    nimg = vall.selection

    for i in nimg:
        
        img, labele = vall[i]     
        bbname = vall.get_image_url(i).split("/")[-1].split(".")[0]             # Extracting image url to retrieve its BB
    
        imwidth, imheight, xmin, ymin, xmax, ymax = miscel.bbinfo(bbname, bdta)

        xminn, yminn, xmaxn, ymaxn = miscel.newloc(imwidth, imheight, xmin, ymin, xmax, ymax)
        start = time.time()
        _, saliency = pgexp.mp(model, img, labele, args.mpiteration)
        end = time.time()
        tm = end - start
        ttotal = ttotal + tm
        l.append(tm)
        z = saliency, labele, bbname
        lslb.append(z)
        xloc,yloc = miscel.findloc(saliency)
        Y = miscel.gtbb(xminn, yminn, xmaxn, ymaxn)
        pg = test.evaluate(Y, (yloc,xloc))
        if pg==1:
            pghits+=1
        elif pg==-1:
            pgmiss+=1
elif args.method == 'mpp':
    model = model.eval()
    # Transforms needs to be applied to our data set
    val_transforms = transforms.Compose([transforms.Resize((224,224))])    
    # 2000 images randomly taken from Imagenet(ILSVRC2012) validation set
    vall = torchray.benchmark.datasets.ImageFolder(dta,transform = val_transforms)
    # number of images we need to calculate things for
    nimg = vall.selection
    iter = [5, 10, 25, 50, 100, 200]
    for j in iter:
        pghits = 0
        pgmiss = 0
        for i in nimg:
            
            img, labele = vall[i]     
            bbname = vall.get_image_url(i).split("/")[-1].split(".")[0]             # Extracting image url to retrieve its BB
        
            imwidth, imheight, xmin, ymin, xmax, ymax = miscel.bbinfo(bbname, bdta)

            xminn, yminn, xmaxn, ymaxn = miscel.newloc(imwidth, imheight, xmin, ymin, xmax, ymax)
            start = time.time()
            _, saliency = pgexp.mp(model, img, labele, j)
            end = time.time()
            tm = end - start
            ttotal = ttotal + tm
            l.append(tm)
            z = saliency, labele, bbname
            lslb.append(z)
            xloc,yloc = miscel.findloc(saliency)
            Y = miscel.gtbb(xminn, yminn, xmaxn, ymaxn)
            pg = test.evaluate(Y, (yloc,xloc))
            if pg==1:
                pghits+=1
            elif pg==-1:
                pgmiss+=1
        print('iterations:  ', j)
        print('hits and miss: ',pghits, pgmiss)

elif args.method == 'epp':
    # Transforms needs to be applied to our data set
    val_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])  
    # 2000 images randomly taken from Imagenet(ILSVRC2012) validation set
    vall = torchray.benchmark.datasets.ImageFolder(dta,transform = val_transforms)
    #maxiter = 100
    #ars = [0.012, 0.025, 0.05, 0.1, 0.25, 0.35, 0.5]
    arss = [[0.05, 0.1], [0.02, 0.08, 0.16],[0.05, 0.15, 0.3],[0.025, 0.05, 0.1, 0.2], [0.012, 0.025, 0.05, 0.1, 0.2], [0.012, 0.025, 0.05, 0.1, 0.2, 0.4],[0.015, 0.03, 0.05, 0.1, 0.25, 0.5],[0.01, 0.04, 0.05, 0.1, 0.2, 0.4]]

    # number of images we need to calculate things for
    nimg = vall.selection
    for j in arss:
        pghits = 0
        pgmiss = 0
        for i in nimg:
            img, labele = vall[i]
            bbname = vall.get_image_url(i).split("/")[-1].split(".")[0]             # Extracting image url to retrieve its BB
            imwidth, imheight, xmin, ymin, xmax, ymax = miscel.bbinfo(bbname,bdta)
            xminn, yminn, xmaxn, ymaxn = miscel.newloc(imwidth, imheight, xmin, ymin, xmax, ymax)
    
            x = img.unsqueeze(0)
            x = x.cuda()
            start = time.time()
            saliency, _ = extremal_perturbation(model, x, labele, reward_func=simple_reward,max_iter=400, debug=False, areas=j ,smooth=0.09,perturbation='blur')
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
        print('iterations:  ', j)
        print('hits and miss: ',pghits, pgmiss)

elif args.method == 'ep':
    # Transforms needs to be applied to our data set
    val_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])  
    # 2000 images randomly taken from Imagenet(ILSVRC2012) validation set
    vall = torchray.benchmark.datasets.ImageFolder(dta,transform = val_transforms)
    #maxiter = 100
    #ars = [0.012, 0.025, 0.05, 0.1, 0.25, 0.35, 0.5]

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
        saliency, _ = extremal_perturbation(model, x, labele, reward_func=simple_reward,max_iter=400, debug=False, areas=[0.012, 0.025, 0.05, 0.1, 0.2],smooth=0.09,perturbation='blur')
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
    val_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])  
    # 2000 images randomly taken from Imagenet(ILSVRC2012) validation set
    vall = torchray.benchmark.datasets.ImageFolder(dta,transform = val_transforms)

    # Make it to be able to get gradient
    '''for p in model.module.parameters():
        p.requires_grad = True'''

    # Initialization of GradCam method
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    # number of images we need to calculate things for
    nimg = vall.selection
    for i in nimg:
        img, labele = vall[i]
        x = img.unsqueeze(0)
        x = x.cuda()
        bbname = vall.get_image_url(i).split("/")[-1].split(".")[0]             # Extracting image url to retrieve its BB
        imwidth, imheight, xmin, ymin, xmax, ymax = miscel.bbinfo(bbname, bdta)
        xminn, yminn, xmaxn, ymaxn = miscel.newloc(imwidth, imheight, xmin, ymin, xmax, ymax)
        start = time.time()
        saliency = cam(input_tensor=x,targets = [ClassifierOutputTarget(labele)])
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

elif args.method == 'cam':
    # Transforms needs to be applied to our data set
    val_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])  
    # 2000 images randomly taken from Imagenet(ILSVRC2012) validation set
    vall = torchray.benchmark.datasets.ImageFolder(dta,transform = val_transforms)
    # number of images we need to calculate things for
    nimg = vall.selection
    for i in nimg:
        img, labele = vall[i]
        x = img.unsqueeze(0)
        x = x.cuda()
        bbname = vall.get_image_url(i).split("/")[-1].split(".")[0]             # Extracting image url to retrieve its BB
        imwidth, imheight, xmin, ymin, xmax, ymax = miscel.bbinfo(bbname,bdta)
        xminn, yminn, xmaxn, ymaxn = miscel.newloc(imwidth, imheight, xmin, ymin, xmax, ymax)
        start = time.time()
        features_blobs = []
        def hook_feature(module, input, output):
            features_blobs.append(output.data.cpu().numpy())
        modulll.register_forward_hook(hook_feature)
        logit = model(x)
        cammask = miscel.bCAM(features_blobs[0], weight_softmax, labele)
        end = time.time()
        tm = end - start
        ttotal = ttotal + tm
        z = cammask, labele, bbname
        lslb.append(z)
        l.append(tm)
        xloc,yloc = miscel.camfindloc(cammask)
        Y = miscel.gtbb(xminn, yminn, xmaxn, ymaxn)
        pg = test.evaluate(Y, (yloc,xloc))
        if pg==1:
            pghits+=1
        elif pg==-1:
            pgmiss+=1

elif args.method == 'ig':
    # Transforms needs to be applied to our data set
    val_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])    
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
        #saliency = torch.mean(attributions, 1,keepdim=True)
        attributions = torch.abs(attributions)
        saliency,_ = torch.max(attributions, 1,keepdim=True)

        end = time.time()
        tm = end - start
        ttotal = ttotal + tm
        z = torch.squeeze(saliency).cpu().numpy(), labele, bbname
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
elif args.method == 'riser':
    # Transforms needs to be applied to our data set
    val_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])    
    # 2000 images randomly taken from Imagenet(ILSVRC2012) validation set
    vall = torchray.benchmark.datasets.ImageFolder(dta,transform = val_transforms)
    nmasks = [ 6000, 7000,7500, 8000, 9000, 10000]
    ssze = [3, 4,5,6,7,8,9]
    # Rise initialization 
    with torch.no_grad():
        for nmsk in nmasks:
            for ssz in ssze:
                pghits = 0
                pgmiss = 0
                # Rise initialization 
                explainer = RISE(model, (224, 224),100)
                explainer.generate_masks(N=nmsk, s=ssz, p1=0.5, savepath='masks.npy')

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
                    saliency = explainer(x)

                    sal = saliency[labele].cpu().numpy()
                    
                    end = time.time()
                    tm = end - start
                    ttotal = ttotal + tm
                    z = sal, labele, bbname
                    lslb.append(z)
                    l.append(tm)
                    #saliency = miscel.normlze(saliency)
                    xloc,yloc = miscel.camfindloc(sal)
                    Y = miscel.gtbb(xminn, yminn, xmaxn, ymaxn)
                    pg = test.evaluate(Y, (yloc,xloc))
                    if pg==1:
                        pghits+=1
                    elif pg==-1:
                        pgmiss+=1
                    del saliency
                    del sal
                print('number of mask and kernSize: ', nmsk, ssz)
                print('Pointing game accuracy: ',pghits/(pghits+pgmiss))

elif args.method == 'rise':
    # Transforms needs to be applied to our data set
    val_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])    
    # 2000 images randomly taken from Imagenet(ILSVRC2012) validation set
    vall = torchray.benchmark.datasets.ImageFolder(dta,transform = val_transforms)

    # Rise initialization 
    explainer = RISE(model, (224, 224), 100)
    explainer.generate_masks(N=7000, s=8, p1=0.5, savepath='masks.npy')

    # number of images we need to calculate things for
    nimg = vall.selection
    with torch.no_grad():

        for i in nimg:
            img, labele = vall[i]
            bbname = vall.get_image_url(i).split("/")[-1].split(".")[0]             # Extracting image url to retrieve its BB
            imwidth, imheight, xmin, ymin, xmax, ymax = miscel.bbinfo(bbname, bdta)
            xminn, yminn, xmaxn, ymaxn = miscel.newloc(imwidth, imheight, xmin, ymin, xmax, ymax)
            x = img.unsqueeze(0)
            x = x.cuda()
            start = time.time()
            saliency = explainer(x)

            sal = saliency[labele].cpu().numpy()
            
            end = time.time()
            tm = end - start
            ttotal = ttotal + tm
            z = sal, labele, bbname
            lslb.append(z)
            l.append(tm)
            #saliency = miscel.normlze(saliency)
            xloc,yloc = miscel.camfindloc(sal)
            Y = miscel.gtbb(xminn, yminn, xmaxn, ymaxn)
            pg = test.evaluate(Y, (yloc,xloc))
            if pg==1:
                pghits+=1
            elif pg==-1:
                pgmiss+=1
            del saliency
            del sal

elif args.method == 'lim':

    def get_preprocess_transform():
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])     
        transf = transforms.Compose([
            transforms.ToTensor(),
            normalize
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
    seg = [10, 20, 30, 40,50]
    for j in seg:
        pghits = 0
        pgmiss = 0
        # Method
        explainer = lime_image.LimeImageExplainer()
        slic_parameters = {'n_segments': j,
                    'compactness': 10,
                    'sigma': 3}
        segmenter = SegmentationAlgorithm('slic', **slic_parameters)

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
                                                    labels = (labele,),
                                                    hide_color=0,
                                                    segmentation_fn=segmenter,
                                                    top_labels= None,  
                                                    num_samples=1000,progress_bar=False)
            sgments = explanation.segments
            hetmap = np.zeros(sgments.shape)
            local_exp = explanation.local_exp
            locexp = local_exp[labele]

            for i, (seg_idx, seg_val) in enumerate(locexp):
                hetmap[sgments == seg_idx] = seg_val

            end = time.time()
            tm = end - start
            ttotal = ttotal + tm
            z = hetmap, labele, bbname
            lslb.append(z)
            l.append(tm)
            xcen = int(np.floor((xmaxn+xminn)/2))
            ycen = int(np.floor((ymaxn+yminn)/2))
            pgalt = hetmap[ycen, xcen]==hetmap.max()
            #saliency = miscel.normlze(saliency)
            xloc,yloc = miscel.camfindloc(hetmap)
            Y = miscel.gtbb(xminn, yminn, xmaxn, ymaxn)
            pg = test.evaluate(Y, (yloc,xloc))
            if pg==1 or pgalt:
                pghits+=1
            else:
                pgmiss+=1
        print('iterations:  ', j)
        print('hits and miss: ',pghits, pgmiss)

elif args.method == 'lime':

    def get_preprocess_transform():
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])     
        transf = transforms.Compose([
            transforms.ToTensor(),
            normalize
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
    slic_parameters = {'n_segments': 40,
                   'compactness': 10,
                   'sigma': 3}
    segmenter = SegmentationAlgorithm('slic', **slic_parameters)

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
                                                labels = (labele,),
                                                hide_color=0,
                                                segmentation_fn=segmenter,
                                                top_labels= None,  
                                                num_samples=1000,progress_bar=False)
        sgments = explanation.segments
        hetmap = np.zeros(sgments.shape)
        local_exp = explanation.local_exp
        locexp = local_exp[labele]

        for i, (seg_idx, seg_val) in enumerate(locexp):
            hetmap[sgments == seg_idx] = seg_val

        end = time.time()
        tm = end - start
        ttotal = ttotal + tm
        z = hetmap, labele, bbname
        lslb.append(z)
        l.append(tm)
        xcen = int(np.floor((xmaxn+xminn)/2))
        ycen = int(np.floor((ymaxn+yminn)/2))
        pgalt = hetmap[ycen, xcen]==hetmap.max()
        #saliency = miscel.normlze(saliency)
        xloc,yloc = miscel.camfindloc(hetmap)
        Y = miscel.gtbb(xminn, yminn, xmaxn, ymaxn)
        pg = test.evaluate(Y, (yloc,xloc))
        if pg==1 or pgalt:
            pghits+=1
        else:
            pgmiss+=1

elif args.method == 'shap':
    # Transforms needs to be applied to our data set
    val_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])    
    # 2000 images randomly taken from Imagenet(ILSVRC2012) validation set
    vall = torchray.benchmark.datasets.ImageFolder(dta,transform = val_transforms)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def normalize(image):
        if image.max() > 1:
            image /= 255
        image = (image - mean) / std
        # in addition, roll the axis so that they suit pytorch
        return torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float().cuda()
    # Method
    X,_ = shap.datasets.imagenet50()
    e = shap.GradientExplainer(model, normalize(X))


    # number of images we need to calculate things for
    nimg = vall.selection
    for i in nimg:
        img, labele = vall[i]
        bbname = vall.get_image_url(i).split("/")[-1].split(".")[0]             # Extracting image url to retrieve its BB
        imwidth, imheight, xmin, ymin, xmax, ymax = miscel.bbinfo(bbname, bdta)
        xminn, yminn, xmaxn, ymaxn = miscel.newloc(imwidth, imheight, xmin, ymin, xmax, ymax)
        x = img.unsqueeze(0)
        x = x.cuda()
        outputs = model(x)
        _, ind = outputs[0].sort(descending=True)
        c = ((ind == labele).nonzero(as_tuple=False)).item()
        start = time.time()
        shap_values,_ = e.shap_values(x, ranked_outputs=c+1, nsamples=100)
        d = torch.from_numpy(shap_values[-1])
        dd = torch.abs(d)

        #saliency = torch.mean(d, 1,keepdim=True)
        saliency,_ = torch.max(dd, 1,keepdim=True)
        end = time.time()
        tm = end - start
        ttotal = ttotal + tm
        z = torch.squeeze(saliency).cpu().numpy(), labele, bbname
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

torch.save(lslb, '/media/mallet/Haniye/cams/optim/'+args.method + '-'+args.model+ '-'+args.data+'.pt')
#l = np.asarray(l)     
#savetxt('/home/mallet/Desktop/Dataa/Runtimes/time'+args.method+args.model+args.data+'.csv', l, delimiter=',')
print('Pointing game accuracy: ',pghits/(pghits+pgmiss))
print('Total time: ',ttotal)
print('hits and misses: ', pghits, pgmiss)
