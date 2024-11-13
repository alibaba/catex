import os

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden

from .imagenet import ImageNet

TO_BE_IGNORED = ["README.txt"]


@DATASET_REGISTRY.register()
class ImageNetA(DatasetBase):
    """ImageNet-A(dversarial).

    This dataset is used for testing only.
    """

    dataset_dir = "imagenet-a"  #imagenet-adversarial
    dataset_name = "imagenet-a"

    def __init__(self, cfg):
        self.full_dir = f'data/{self.dataset_dir[:-2]}/images/val'
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")  # imagenet-a

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)

        data = self.read_data(classnames)

        super().__init__(train_x=data, test=data)
        self._num_classes = len(self.valid_classes)
        folders = sorted(os.listdir(self.full_dir))
        self._classnames = [classnames[f] for f in folders]


    def read_data(self, classnames):
        import torch
        image_dir = self.image_dir
        folders = listdir_nohidden(self.full_dir, sort=True)
        folders = [f for f in folders if f not in TO_BE_IGNORED]
        items = []

        self.valid_classes = torch.full((len(folders),), False)
        for label, folder in enumerate(folders):
            data_dir = os.path.join(image_dir, folder)
            if not os.path.exists(data_dir):
                continue
            self.valid_classes[label] = True

            imnames = listdir_nohidden(data_dir)
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
