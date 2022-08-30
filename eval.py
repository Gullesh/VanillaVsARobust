
import torch
import torchvision.transforms as transforms
import os, argparse
from tqdm import tqdm
import torchvision.datasets as datasets_torch
from robustness.datasets import ImageNet
from load_model import load_adv_prop_model
import torchvision

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,), exact=False):
    """
        Computes the top-k accuracy for the specified values of k

        Args:
            output (ch.tensor) : model output (N, classes) or (N, attributes) 
                for sigmoid/multitask binary classification
            target (ch.tensor) : correct labels (N,) [multiclass] or (N,
                attributes) [multitask binary]
            topk (tuple) : for each item "k" in this tuple, this method
                will return the top-k accuracy
            exact (bool) : whether to return aggregate statistics (if
                False) or per-example correctness (if True)

        Returns:
            A list of top-k accuracies.
    """
    with torch.no_grad():
        # Binary Classification
        if len(target.shape) > 1:
            assert output.shape == target.shape, \
                "Detected binary classification but output shape != target shape"
            return [torch.round(torch.sigmoid(output)).eq(torch.round(target)).float().mean()], [-1.0] 

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        res_exact = []
        for k in topk:
            correct_k = correct[:k].view(-1).float() if k == 1 else correct.sum(axis=0, dtype=float)
            ck_sum = correct_k.sum(0, keepdim=True)
            res.append(ck_sum.mul_(100.0 / batch_size))
            res_exact.append(correct_k)

        if not exact:
            return res
        else:
            return res_exact


def evaluate(args):
    model_name = args.model_name
    batch_size = args.batch_size
    madry_model= args.madry_model
    data_path  = args.data_path
    if args.correct_only:
        correct_sample_list = torch.load('resnet50_pgd5_imagenet_intersect.pth')

    if not args.adv_prop or madry_model:
        import torchvision.models as models
        if model_name == 'googlenet':
            model = models.googlenet(pretrained=True)
        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
        model.cuda()
        model.eval()
    else:

        model = load_adv_prop_model(args.model_path)  
        model.eval()
        model.cuda()

    if madry_model:
        dataset = ImageNet(data_path)
        train_loader, val_loader = dataset.make_loaders(batch_size=batch_size, workers=8)
        dataloader = val_loader
    else:
        dataloader = get_data_loader(
            args.resize, args.normalize, data_path, batch_size, args.imagenetc or args.imagenetc_all, args.adv_prop
        )

    if args.save_correct_samples:
        correct_samples = []

    with torch.no_grad():
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')


        pbar = tqdm(total=len(dataloader))
        pbar.set_description(f"Classifying {model_name}")
        for i, (img, target) in enumerate(dataloader):
            if args.correct_only:
                sample_ids = dataloader.dataset.samples[i*batch_size: (i+1)*batch_size]
                fliter_mask = drop_incorrect_samples(sample_ids,correct_sample_list)
                img = img[fliter_mask]
                target = target[fliter_mask]

            target_tensor = torch.tensor(list(target)).to('cuda')
            img = img.to('cuda')
            logit = model(img, target)[0] if args.adv_prop else model(img)
            prob = torch.nn.functional.softmax(logit, dim=1)
            (acc1, acc5) = accuracy(prob, target_tensor, (1, 5))

            top1.update(acc1[0], n=len(target))
            top5.update(acc5[0], n=len(target))


            # to retrive file names
            if args.save_correct_samples:
                samples = dataloader.dataset.samples[i*batch_size: (i+1)*batch_size]
                result = accuracy(prob, target_tensor, (1,), exact=True)
                correct_file_names = get_correct_samples(samples, result[0])
                correct_samples.extend(correct_file_names)

            # Display progress
            pbar.update(1)
        pbar.close()

    if args.save_correct_samples:
        torch.save(correct_samples, f'{model_name}_correct_samples.pth' if not args.adv_prop else f'{model_name}_adv_prop_correct_samples.pth')

    if args.imagenetc or args.imagenetc_all:
        model_prefix = 'standard' if not args.adv_prop else 'adv_prop'
        print(
            f'{model_prefix}_{model_name} on {args.distortion}: Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
                top1=top1, top5=top5
            )
        )

    else: 
        print(f'{model_name}: Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

def drop_incorrect_samples(samples,correct_samples):
    correct_samples_set = set(correct_samples)
    new_samples = []
    for sample in samples:
        if sample[0].split('/')[-1] in correct_samples_set:
            new_samples.append(True)
        else:
            new_samples.append(False)
    return new_samples

def get_correct_samples(samples, results):
    """
    Save correct samples
    """
    img_ids = [sample[0].split('/')[-1] for sample in samples]
    return [img_ids[i] for i, label in enumerate(results) if label == 1]

def get_data_loader(resize, normalizing, data_path, batch_size, imagenet_c, adv_prop=False):
    if adv_prop:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    trans = []
    if resize:
        trans.append(transforms.Resize(256))
        trans.append(transforms.CenterCrop(224))
    trans.append(transforms.ToTensor())
    if normalizing:
        trans.append(normalize)

    folder_path = os.path.join(data_path, 'val') if not imagenet_c else data_path
    return torch.utils.data.DataLoader(
            datasets_torch.ImageFolder(folder_path, transforms.Compose(trans)),
            batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=False)

def argParser():
    parser = argparse.ArgumentParser(description='Evaluate models on ImageNet.')
    parser.add_argument('--model_name', default='resnet50',  help='Model name')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for dataloader.')
    parser.add_argument('--model_path', default='pgd_1.pth.tar', help='Model checkpoint. This argument is not necessary for standard models.')
    parser.add_argument('--data_path', default='/home/peijie/dataset/ILSVRC2012', help='path to dataset')
    parser.add_argument('--resize', default=True, type=bool, help='resize or not')
    parser.add_argument('--x', default=True, type=bool, help='normalize or not')
    parser.add_argument('--madry_model', action='store_true', help='weather it is a madry model')
    parser.add_argument('--adv_prop', action='store_true', help='weather it is a adv prop model')
    parser.add_argument('--imagenetc', action='store_true')
    parser.add_argument('--imagenetc_all', action='store_true')
    parser.add_argument('--distortion', default='')
    parser.add_argument('--save_correct_samples', action='store_true')
    parser.add_argument('--correct_only', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = argParser()
    if args.imagenetc_all:
        types = os.listdir('/home/peijie/dataset/imagenet_C')
        for type in types:
            args.distortion = type
            args.data_path = '/home/peijie/dataset/imagenet_C/' + type + '/3'
            evaluate(args)
    else:
        evaluate(args)
