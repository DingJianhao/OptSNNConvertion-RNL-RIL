from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os

num_classes = {'cifar10': 10, 'cifar100': 100, 'mnist':10, 'imagenet':1000}
datapath = {
    'cifar10': 'G:/dataset/cifar10',
    'cifar100': 'G:/dataset/cifar100',
    'mnist':'G:/dataset/mnist',
    'imagenet': '/gdata/ImageNet2012'
}


def print_args(args):
    print('ARGUMENTS:')
    for arg in vars(args):
        print(f">>> {arg}: {getattr(args, arg)}")

def load_cv_data(data_aug, batch_size, workers, dataset, data_target_dir):
    if dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif dataset == 'mnist':
        mean = (0.1307,)
        std = (0.3081,)
    elif dataset == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        assert False, f"Unknown dataset : {dataset}"

    if data_aug:
        if dataset == 'svhn':
            train_transform = transforms.Compose([
                            transforms.RandomCrop(32, padding=2),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
            test_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
        elif dataset == 'imagenet':
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            test_transform = transforms.Compose([
                transforms.Resize(int(224 / 0.875)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            train_transform = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(32, padding=2),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
            test_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
    else:
        if dataset == 'imagenet':
            train_transform = transforms.Compose([
                transforms.Resize(int(224 / 0.875)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            test_transform = transforms.Compose([
                transforms.Resize(int(224 / 0.875)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            train_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)
                            ])
            test_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)
                            ])
    if dataset == 'cifar10':
        train_data = datasets.CIFAR10(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(data_target_dir, train=False, transform=test_transform, download=True)
    elif dataset == 'cifar100':
        train_data = datasets.CIFAR100(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(data_target_dir, train=False, transform=test_transform, download=True)
    elif dataset == 'svhn':
        train_data = datasets.SVHN(data_target_dir, split='train', transform=train_transform, download=True)
        test_data = datasets.SVHN(data_target_dir, split='test', transform=test_transform, download=True)
    elif dataset == 'mnist':
        train_data = datasets.MNIST(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.MNIST(data_target_dir, train=False, transform=test_transform, download=True)
    elif dataset == 'imagenet':
        train_data = datasets.ImageFolder(root=os.path.join(data_target_dir, 'train'),transform=train_transform)
        test_data = datasets.ImageFolder(root=os.path.join(data_target_dir, 'val'),transform=test_transform)
    else:
        assert False, 'Do not support dataset : {}'.format(dataset)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    return train_dataloader, test_dataloader


if __name__ == "__main__":
    import models
    import modules
    import torch
    import numpy as np
    import torch.nn as nn

    train_dataloader, test_dataloader = load_cv_data(data_aug=False,
                                                     batch_size=50,
                                                     workers=0,
                                                     dataset='cifar10',
                                                     data_target_dir=datapath['cifar10']
                                                     )
    percentage = 0.004  # load 0.004 of the data
    norm_data_list = []
    for idx, (imgs, targets) in enumerate(train_dataloader):
        norm_data_list.append(imgs)
        if idx == int(len(train_dataloader) * percentage) - 1:
            break
    norm_data = torch.cat(norm_data_list)



    model = models.__dict__['vgg16'](num_classes=10, dropout=0.1)
    model = modules.replace_maxpool2d_by_avgpool2d(model)
    model = modules.replace_relu_by_spikingnorm(model, True)
    v = []
    model.load_state_dict(torch.load("C:/Users/dell/Desktop/Projects/Paper1_Figures/Figure6/vgg16_cifar10_[0.100_91.340_1.801].pth")['net'])
    cnt = 0
    for m in model.modules():
        if isinstance(m,modules.SpikingNorm):
            v.append(m.calc_v_th().data.item())
            cnt+=1
    x = model(norm_data)
    v.append(torch.max(x).item())
    print(len(v))
    np.savetxt('vth_rnl.csv', np.array(v), delimiter=',')

    model = modules.replace_spikingnorm_by_relu(model)
    model.load_state_dict(torch.load("C:/Users/dell/Desktop/Projects/Paper1_Code/normal/pretrain/vgg16_cifar10.pth")['net'])



    max_activation = []
    def hook(module,input,output):
        # print(torch.max(output))
        max_activation.append(torch.max(output).item())

    for m in model.modules():
        if isinstance(m,nn.ReLU):
            m.register_forward_hook(hook)

    x = model(norm_data)
    max_activation.append(torch.max(x).item())
    print(len(max_activation))
    np.savetxt('vth_original.csv', np.array(max_activation), delimiter=',')

    np.savetxt('vth_x0.8.csv', np.array(max_activation) * 0.8, delimiter=',')


