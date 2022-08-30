import torch
import net
import torch.nn as nn

class NoOpAttacker():
    
    def attack(self, image, label, model):
        return image, -torch.ones_like(label)

class MixBatchNorm2d(nn.BatchNorm2d):
    '''
    if the dimensions of the tensors from dataloader is [N, 3, 224, 224]
    that of the inputs of the MixBatchNorm2d should be [2*N, 3, 224, 224].

    If you set batch_type as 'mix', this network will using one batchnorm (main bn) to calculate the features corresponding to[:N, 3, 224, 224],
    while using another batch normalization (auxiliary bn) for the features of [N:, 3, 224, 224].
    During training, the batch_type should be set as 'mix'.

    During validation, we only need the results of the features using some specific batchnormalization.
    if you set batch_type as 'clean', the features are calculated using main bn; if you set it as 'adv', the features are calculated using auxiliary bn.

    Usually, we use to_clean_status, to_adv_status, and to_mix_status to set the batch_type recursively. It should be noticed that the batch_type should be set as 'adv' while attacking.
    '''
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(MixBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.aux_bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine,
                                     track_running_stats=track_running_stats)
        self.batch_type = 'clean'

    def forward(self, input):
        if self.batch_type == 'adv':
            input = self.aux_bn(input)
        elif self.batch_type == 'clean':
            input = super(MixBatchNorm2d, self).forward(input)
        else:
            assert self.batch_type == 'mix'
            batch_size = input.shape[0]
            # input0 = self.aux_bn(input[: batch_size // 2])
            # input1 = super(MixBatchNorm2d, self).forward(input[batch_size // 2:])
            input0 = super(MixBatchNorm2d, self).forward(input[:batch_size // 2])
            input1 = self.aux_bn(input[batch_size // 2:])
            input = torch.cat((input0, input1), 0)
        return input


def load_adv_prop_model(checkpoint='pgd_1.pth.tar'):
    mixbn = True
    attacker = NoOpAttacker()
    checkpoint = torch.load(checkpoint)
    arch = 'resnet50'

    if mixbn:
        norm_layer = MixBatchNorm2d
    else:
        norm_layer = None
    model = net.__dict__[arch](num_classes=1000, norm_layer=norm_layer)
    model.set_attacker(attacker)
    model.set_mixbn(mixbn)
    model = torch.nn.DataParallel(model).cuda()


    if mixbn:
        to_merge = {}
        for key in checkpoint['state_dict']:
            if 'bn' in key:
                tmp = key.split("bn")
                aux_key = tmp[0] + 'bn' + tmp[1][0] + '.aux_bn' + tmp[1][1:]
                to_merge[aux_key] = checkpoint['state_dict'][key]
            elif 'downsample.1' in key:
                tmp = key.split("downsample.1")
                aux_key = tmp[0] + 'downsample.1.aux_bn' + tmp[1]
                to_merge[aux_key] = checkpoint['state_dict'][key]
        checkpoint['state_dict'].update(to_merge)
    # model = torch.nn.DataParallel(model).cuda()
    unexpceted_keys = ["module.bn1.aux_bn.aux_", "module.layer1.0.bn1.aux_bn.aux_", "module.layer1.0.bn2.aux_bn.aux_", "module.layer1.0.bn3.aux_bn.aux_", "module.layer1.0.downsample.1.aux_bn..aux_bnweight", "module.layer1.0.downsample.1.aux_bn..aux_bnbias", "module.layer1.0.downsample.1.aux_bn..aux_bnrunning_mean", "module.layer1.0.downsample.1.aux_bn..aux_bnrunning_var", "module.layer1.0.downsample.1.aux_bn..aux_bnnum_batches_tracked", "module.layer1.1.bn1.aux_bn.aux_", "module.layer1.1.bn2.aux_bn.aux_", "module.layer1.1.bn3.aux_bn.aux_", "module.layer1.2.bn1.aux_bn.aux_", "module.layer1.2.bn2.aux_bn.aux_", "module.layer1.2.bn3.aux_bn.aux_", "module.layer2.0.bn1.aux_bn.aux_", "module.layer2.0.bn2.aux_bn.aux_", "module.layer2.0.bn3.aux_bn.aux_", "module.layer2.0.downsample.1.aux_bn..aux_bnweight", "module.layer2.0.downsample.1.aux_bn..aux_bnbias", "module.layer2.0.downsample.1.aux_bn..aux_bnrunning_mean", "module.layer2.0.downsample.1.aux_bn..aux_bnrunning_var", "module.layer2.0.downsample.1.aux_bn..aux_bnnum_batches_tracked", "module.layer2.1.bn1.aux_bn.aux_", "module.layer2.1.bn2.aux_bn.aux_", "module.layer2.1.bn3.aux_bn.aux_", "module.layer2.2.bn1.aux_bn.aux_", "module.layer2.2.bn2.aux_bn.aux_", "module.layer2.2.bn3.aux_bn.aux_", "module.layer2.3.bn1.aux_bn.aux_", "module.layer2.3.bn2.aux_bn.aux_", "module.layer2.3.bn3.aux_bn.aux_", "module.layer3.0.bn1.aux_bn.aux_", "module.layer3.0.bn2.aux_bn.aux_", "module.layer3.0.bn3.aux_bn.aux_", "module.layer3.0.downsample.1.aux_bn..aux_bnweight", "module.layer3.0.downsample.1.aux_bn..aux_bnbias", "module.layer3.0.downsample.1.aux_bn..aux_bnrunning_mean", "module.layer3.0.downsample.1.aux_bn..aux_bnrunning_var", "module.layer3.0.downsample.1.aux_bn..aux_bnnum_batches_tracked", "module.layer3.1.bn1.aux_bn.aux_", "module.layer3.1.bn2.aux_bn.aux_", "module.layer3.1.bn3.aux_bn.aux_", "module.layer3.2.bn1.aux_bn.aux_", "module.layer3.2.bn2.aux_bn.aux_", "module.layer3.2.bn3.aux_bn.aux_", "module.layer3.3.bn1.aux_bn.aux_", "module.layer3.3.bn2.aux_bn.aux_", "module.layer3.3.bn3.aux_bn.aux_", "module.layer3.4.bn1.aux_bn.aux_", "module.layer3.4.bn2.aux_bn.aux_", "module.layer3.4.bn3.aux_bn.aux_", "module.layer3.5.bn1.aux_bn.aux_", "module.layer3.5.bn2.aux_bn.aux_", "module.layer3.5.bn3.aux_bn.aux_", "module.layer4.0.bn1.aux_bn.aux_", "module.layer4.0.bn2.aux_bn.aux_", "module.layer4.0.bn3.aux_bn.aux_", "module.layer4.0.downsample.1.aux_bn..aux_bnweight", "module.layer4.0.downsample.1.aux_bn..aux_bnbias", "module.layer4.0.downsample.1.aux_bn..aux_bnrunning_mean", "module.layer4.0.downsample.1.aux_bn..aux_bnrunning_var", "module.layer4.0.downsample.1.aux_bn..aux_bnnum_batches_tracked", "module.layer4.1.bn1.aux_bn.aux_", "module.layer4.1.bn2.aux_bn.aux_", "module.layer4.1.bn3.aux_bn.aux_", "module.layer4.2.bn1.aux_bn.aux_", "module.layer4.2.bn2.aux_bn.aux_", "module.layer4.2.bn3.aux_bn.aux_"]
    for key in unexpceted_keys:
        del checkpoint['state_dict'][key]
    model.load_state_dict(checkpoint['state_dict'])
    return model
