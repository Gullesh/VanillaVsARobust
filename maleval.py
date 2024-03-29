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
import torch.nn.functional as F
import alx
import alexp
from torchray.attribution.extremal_perturbation import extremal_perturbation, contrastive_reward, simple_reward
from pytorch_grad_cam import GradCAM
from captum.attr import IntegratedGradients
from torchray.attribution.rise import rise
from lime import lime_image
import shap
from RISE.explanations import RISE
from RISE.utils import *
from lime.wrappers.scikit_image import SegmentationAlgorithm
from RISE.evaluation import CausalMetric, auc, gkern
import torch.backends.cudnn as cudnn
from torch.nn.functional import conv2d
import torch.nn as nn
import numpy
from scipy.stats.stats import pearsonr   

cudnn.benchmark = True

model_options = ['resnet50', 'resnet50r', 'googlenet', 'googlenetr','vgg16', 'alexnet', 'alexnetr','densenet','mobile', 'densenetr','mobiler']
method_options = ['gradient', 'mp','ep', 'gradcam', 'rise','ig', 'rise', 'lime','shap','riser', 'cam']
data_options = ['rnddata', 'rsnt', 'ggl', 'alx', 'subalx', 'subrsnt', 'subggl','mbl','dens']

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
    model.eval()
    target_layer = model.model.features[11]
elif args.model == 'densenet':
    model = alx.loaddens()
    target_layers = [model.features[-1]]
    modulll = model._modules.get('features')
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())
elif args.model == 'densenetr':
    model = alx.loaddensr()
    target_layers = [model.model.model.features[-1]]
    modulll = model._modules.get('model')._modules.get('model')._modules.get('features')
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())
elif args.model == 'mobile':
    model = alx.loadmob()
    target_layers = [model.features[18]]
elif args.model == 'mobiler':
    model = alx.loadmobr()
    target_layers = [model.model.model.features[18]]
else:
    print('Error: please choose a valid model')


if args.data == 'rnddata':
    dta = '/home/mallet/Desktop/Dataa/rnddata/valid'
    bdta = '/home/mallet/Desktop/Dataa/rnddata/val/'
elif args.data == 'alx':
    dta = '/home/mallet/Desktop/Dataa/alx/valid'
    bdta = '/home/mallet/Desktop/Dataa/alx/val/'
elif args.data == 'subalx':
    dta = '/home/mallet/Desktop/Dataa/subalx/valid'
    bdta = '/home/mallet/Desktop/Dataa/subalx/val/'
elif args.data == 'mbl':
    dta = '/home/mallet/Desktop/Dataa/mbl/valid'
    bdta = '/home/mallet/Desktop/Dataa/mbl/val/'
elif args.data == 'dens':
    dta = '/home/mallet/Desktop/Dataa/dens/valid'
    bdta = '/home/mallet/Desktop/Dataa/dens/val/'
else:
    print('Error: please choose a valid data')

tol = args.tol

#lists for confidences and metrics
confid = []
metri1 = []
metri2 = []

# Parameters to use for insertion and deletion
klen = 11
ksig = 5
kern = gkern(klen, ksig)

# Function that blurs input image
blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2)

insertion = CausalMetric(model, 'ins', 224*8, substrate_fn=blur)
deletion = CausalMetric(model, 'del', 224*8, substrate_fn=torch.zeros_like)
delvals = []
inservals = []

if args.method == 'gradient':
    # Transforms needs to be applied to our data set
    val_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])    
    # 2000 images randomly taken from Imagenet(ILSVRC2012) validation set
    vall = torchray.benchmark.datasets.ImageFolder(dta,transform = val_transforms)
    # number of images we need to calculate things for
    nimg = vall.selection
    for i in nimg:
        img, labele = vall[i]
        yy = img.unsqueeze(0)
        y = yy.cuda()
        prob = model(y)
        conf, predicted = torch.max(prob, 1)
        confid.append(conf.item())
        print(torch.max(prob, 1))

        saliency = gradient(model, y, predicted.item())
        saliency = torch.squeeze(saliency).cpu().detach().numpy()
        sal = miscel.normlze(saliency)
        deleti = deletion.single_run(yy, sal)
        inseri = insertion.single_run(yy, sal)
        delvals.append(auc(deleti))
        inservals.append(auc(inseri))


elif args.method == 'mp':

    # Transforms needs to be applied to our data set
    val_transforms = transforms.Compose([transforms.Resize((224,224))])    
    # 2000 images randomly taken from Imagenet(ILSVRC2012) validation set
    vall = torchray.benchmark.datasets.ImageFolder(dta,transform = val_transforms)
    # number of images we need to calculate things for
    nimg = vall.selection
    contsor = transforms.ToTensor()

    for i in nimg:
        
        img, labele = vall[i]     
   
        img, labele = vall[i]     
        yy = contsor(img).unsqueeze(0)
        y = yy.cuda()
        prob = model(y)
        conf, predicted = torch.max(prob, 1)
        confid.append(conf.item())

        _, saliency = alexp.mp(model, img, predicted.item(), args.mpiteration)
        saliency = torch.squeeze(saliency).cpu().detach().numpy()
        #saliency = miscel.normlze(saliency)
        deleti = deletion.single_run(yy, saliency)
        inseri = insertion.single_run(yy, saliency)
        delvals.append(auc(deleti))
        inservals.append(auc(inseri))
        

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
        saliency, _ = extremal_perturbation(model, x, labele, reward_func=simple_reward, debug=False, areas=[0.025, 0.05, 0.1],smooth=0.09,perturbation='blur')
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

#[0.025, 0.05, 0.1, 0.2]

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
        yy = img.unsqueeze(0)
        y = yy.cuda()
        prob = model(y)
        conf, predicted = torch.max(prob, 1)
        confid.append(conf.item())
        saliency = cam(input_tensor=y, target_category=predicted.item())
        saliency = np.squeeze(saliency)
        saliency = miscel.normlze(saliency)
        deleti = deletion.single_run(yy, saliency)
        inseri = insertion.single_run(yy, saliency)
        delvals.append(auc(deleti))
        inservals.append(auc(inseri))
        
elif args.method == 'cam':
    # Transforms needs to be applied to our data set
    val_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])  
    # 2000 images randomly taken from Imagenet(ILSVRC2012) validation set
    vall = torchray.benchmark.datasets.ImageFolder(dta,transform = val_transforms)
    # number of images we need to calculate things for
    nimg = vall.selection
    for i in nimg:
        img, labele = vall[i]
        yy = img.unsqueeze(0)
        y = yy.cuda()
        prob = model(y)
        conf, predicted = torch.max(prob, 1)
        confid.append(conf.item())
        features_blobs = []
        def hook_feature(module, input, output):
            features_blobs.append(output.data.cpu().numpy())
        modulll.register_forward_hook(hook_feature)
        logit = model(y)
        cammask = miscel.bCAM(features_blobs[0], weight_softmax, labele)
        saliency = cv2.resize(cammask, (224, 224))
        saliency = miscel.normlze(saliency)
        deleti = deletion.single_run(yy, saliency)
        inseri = insertion.single_run(yy, saliency)
        delvals.append(auc(deleti))
        inservals.append(auc(inseri))
        
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

    # Rise initialization 
    explainer = RISE(model, (224, 224),100)
    explainer.generate_masks(N=nmsk, s=ssz, p1=0.5, savepath='masks.npy')

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
            saliency = explainer(x)

            sal = saliency[labele].cpu().numpy()
            
            #saliency = miscel.normlze(saliency)
            xloc,yloc = miscel.camfindloc(sal)
            Y = miscel.gtbb(xminn, yminn, xmaxn, ymaxn)
            pg = test.evaluate(Y, (yloc,xloc))
            if pg==1:
                pghits+=1
            elif pg==-1:
                pgmiss+=1
            del sal
            del saliency

elif args.method == 'riser':
    # Transforms needs to be applied to our data set
    val_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])    
    # 2000 images randomly taken from Imagenet(ILSVRC2012) validation set
    vall = torchray.benchmark.datasets.ImageFolder(dta,transform = val_transforms)
    nmasks = [ 7000, 7500, 8000, 8500]
    ssze = [3, 4, 5, 6, 7, 8]
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
                    saliency = explainer(x)

                    sal = saliency[labele].cpu().numpy()
                    
                    #saliency = miscel.normlze(saliency)
                    xloc,yloc = miscel.camfindloc(sal)
                    Y = miscel.gtbb(xminn, yminn, xmaxn, ymaxn)
                    pg = test.evaluate(Y, (yloc,xloc))
                    if pg==1:
                        pghits+=1
                    elif pg==-1:
                        pgmiss+=1
                    del sal
                    del saliency
                print('number of mask and kernSize: ', nmsk, ssz)
                print('Pointing game accuracy: ',pghits/(pghits+pgmiss))

elif args.method == 'lime':

    def get_preprocess_transform():    
        transf = transforms.Compose([
            transforms.ToTensor()])    

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
    slic_parameters = {'n_segments': 50,
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
        elif pg==-1:
            pgmiss+=1


elif args.method == 'shap':
    # Transforms needs to be applied to our data set
    val_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])    
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
        saliency = torch.mean(d, 1,keepdim=True)
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
    

inserr = numpy.array(inservals)
delett = numpy.array(delvals)

#confid = confid.cpu().detach().numpy()
print('Mean and std of insertion game: ', "%.4f" % (numpy.mean(inserr)) +u"\u00B1"+"%.4f" % (numpy.std(inserr)))
print('Mean and std of deletion game: ',"%.4f" % (numpy.mean(delett))  +u"\u00B1"+"%.4f" % (numpy.std(delett)))
insercor, ـ = pearsonr(inservals, confid)
delcor, ـ = pearsonr(delvals, confid)
print('correlation and p-value of insertion and confidence: ', "%.2f" % insercor)
print('correlation and p-value of deletion and confidence: ', "%.2f" % delcor)


