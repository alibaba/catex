# Copyright (c) Alibaba, Inc. and its affiliates.
import os, ast
import numpy as np
from PIL import Image

import torch
import torchvision
from torch.utils.data import DataLoader, Subset, dataloader
from torchvision import transforms


class CLIPFeatDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, epoch=0, nshot=-1, split='train'):
        super().__init__()
        self.data_dir = data_dir
        self.nshot = nshot
        self.split = split
        self.load_data(epoch)
        self.epoch = epoch

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.paths is None:  # for OOD 
            return self.features[idx], self.targets[idx]
        else:
            output = {
                'img': self.features[idx],
                "label": self.targets[idx],
                "domain": '',
                "impath": self.paths[idx],
                "index": idx
            }
            return output 

    def load_data(self, epoch=None):
        if hasattr(self, 'epoch') and epoch == self.epoch:
            return
        prefix = ('' if epoch is None else f'ep{epoch}_') + self.split
        if self.nshot > 0:
            prefix = f'{self.nshot}shot_' + prefix
        self.features = torch.load(f'{self.data_dir}/{prefix}_image_features.pt', map_location='cpu')
        self.targets = torch.load(f'{self.data_dir}/{prefix}_labels.pt', map_location='cpu').long()

        path_file = f'{self.data_dir}/{prefix}_paths.txt'
        if os.path.exists(path_file):
            with open(path_file, 'r') as f:
                self.paths = f.read().splitlines()
            self.epoch = epoch
            if 'train' in prefix:
                print(f'\nLoaded dataset from epoch {epoch}\n')
        else:
            self.paths = None


class LargeOODDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, id_name, ood_name, transform):
        ood_name_dict = {'texture': 'dtd', 'inaturalist': 'iNaturalist', 'places': 'Places', 'sun': 'SUN'}
        data_root = f'{root}/{ood_name_dict[ood_name]}'
        target_transform = lambda x: x * 0 - 1  # all -1

        super().__init__(data_root, transform, target_transform=target_transform)
        print("LargeOODDataset (id %s, ood %s) Contain %d images" % (id_name, ood_name, len(self)))


class SemanticOODDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, id_name, ood_name, transform):
        assert 'imagenet' in id_name and ood_name in ['easy', 'rand', 'hard']
        
        data_root = f'{root}/{id_name}/ood_{ood_name}'
        target_transform = lambda x: x * 0 - 1  # all -1

        super().__init__(data_root, transform, target_transform=target_transform)
        print("SemanticOODDataset (id %s, ood %s) Contain %d images" % (id_name, ood_name, len(self)))


class ClassOODDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, id_name, ood_name, transform):
        assert 'imagenet' in id_name
        
        if 'severity' in ood_name:
            data_root = f'{root}/{id_name}/{ood_name}'
        else:
            assert 'imagenet' in ood_name and '-o' in ood_name
            data_root = f'{root}/../{ood_name}/images'
        target_transform = lambda x: x * 0 - 1  # all -1

        super().__init__(data_root, transform, target_transform=target_transform)
        print("ClassOODDataset (id %s, ood %s) Contain %d images" % (id_name, ood_name, len(self)))


class SCOODDataset(torch.utils.data.Dataset):

    def __init__(self, root, id_name, ood_name, transform):

        super(SCOODDataset, self).__init__()

        assert id_name in ['cifar10', 'cifar100', 'imagenet', 'imagenet100']
        if 'imagenet' in id_name:
            id_name = 'cifar10'
        if ood_name == 'cifar':
            assert 'cifar' in id_name
            if id_name == 'cifar10':
                ood_name = 'cifar100'
            else:
                ood_name = 'cifar10'

        imglist_path = os.path.join(root, 'data/imglist/benchmark_%s' % id_name, 'test_%s.txt' % ood_name)

        with open(imglist_path) as fp:
            self.imglist = fp.readlines()
        
        self.transform = transform
        self.root = root

        print("SCOODDataset (id %s, ood %s) Contain %d images" % (id_name, ood_name, len(self.imglist)))

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        # parse the string in imglist file:
        line = self.imglist[index].strip("\n")
        tokens = line.split(" ", 1)
        image_name, extra_str = tokens[0], tokens[1]
        extras = ast.literal_eval(extra_str)
        sc_label = extras['sc_label'] # the ood label is here. -1 means ood.

        # read image according to image name:
        img_path = os.path.join(self.root, 'data', 'images', image_name)
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, sc_label


class TinyImages(torch.utils.data.Dataset):

    def __init__(self, root, transform):

        super(TinyImages, self).__init__()

        self.data = np.load(os.path.join(root, 'tinyimages80m', '300K_random_images.npy'))
        self.transform = transform

        print("TinyImages Contain {} images".format(len(self.data)))

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, -1  # -1 is the class

    def __len__(self):
        return len(self.data)


def tinyimages300k_dataloaders(num_samples=300000, train_batch_size=64, num_workers=8, data_root_path='/ssd1/haotao/datasets'):

    num_samples = int(num_samples)

    data_dir = os.path.join(data_root_path)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    train_set = Subset(TinyImages(data_dir, train=True, transform=train_transform, download=True), list(range(num_samples)))

    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers,
                                drop_last=True, pin_memory=True)

    return train_loader
    

class _RepeatSampler:
    """ Sampler that repeats forever
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class InfiniteDataLoader(dataloader.DataLoader):
    """ Dataloader that reuses workers
    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)

