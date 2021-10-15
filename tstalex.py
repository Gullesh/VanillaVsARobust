import alx
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


val_transforms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])    
val_dataset = torchvision.datasets.ImageFolder('/home/mallet/Downloads/imgnet/val',transform = val_transforms)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=500, num_workers=8)

model = alx.alexr()

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in val_loader:
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        # calculate outputs by running images through the network
        outputs= model(images)
        print(outputs.shape)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)   

        total += labels.size(0)
        if (total%5000)==0:
            print('5k is done!')
        correct += (predicted == labels).sum().item()
    print(correct/total)