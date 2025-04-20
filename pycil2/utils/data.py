import numpy as np
from torchvision import datasets, transforms
from pycil2.utils.toolkit import split_images_labels
from . import autoaugment
from . import ops

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None
    
    def __init__(self, cls_seq=None):
        if cls_seq is None:
            raise ValueError("cls_seq cannot be None, it must be a sequence of class labels")
        self.class_order = cls_seq


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]
    
    def __init__(self, cls_seq=None):
        if cls_seq is None:
            cls_seq = np.arange(10).tolist()
        if len(cls_seq) != 10:
            raise ValueError(f"CIFAR10 requires 10 classes, but got {len(cls_seq)}")
        super().__init__(cls_seq)

    def download_data(self, download_path="./data"):
        #print(f'download path : {download_path}')
        train_dataset = datasets.cifar.CIFAR10(root=download_path, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10(root=download_path, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]
    
    def __init__(self, cls_seq=None):
        if cls_seq is None:
            cls_seq = np.arange(100).tolist()
        if len(cls_seq) != 100:
            raise ValueError(f"CIFAR100 requires 100 classes, but got {len(cls_seq)}")
        super().__init__(cls_seq)

    def download_data(self, download_path="./data"):
        #print(f'download path : {download_path}')
        
        train_dataset = datasets.cifar.CIFAR100(root=download_path, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(root=download_path, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100_AA(iCIFAR100):
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        autoaugment.CIFAR10Policy(),
        transforms.ToTensor(),
        ops.Cutout(n_holes=1, length=16),
    ]
    
    def __init__(self, cls_seq=None):
        super().__init__(cls_seq)


class iCIFAR10_AA(iCIFAR10):
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        autoaugment.CIFAR10Policy(),
        transforms.ToTensor(),
        ops.Cutout(n_holes=1, length=16),
    ]
    
    def __init__(self, cls_seq=None):
        super().__init__(cls_seq)


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    
    def __init__(self, cls_seq=None):
        if cls_seq is None:
            cls_seq = np.arange(1000).tolist()
        if len(cls_seq) != 1000:
            raise ValueError(f"ImageNet1000 requires 1000 classes, but got {len(cls_seq)}")
        super().__init__(cls_seq)

    def download_data(self, download_path="./data"):
        # assert download_path != "./data", "You should specify the folder of your ImageNet dataset"
        train_dir = f"{download_path}/train/"
        test_dir = f"{download_path}/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    
    def __init__(self, cls_seq=None):
        if cls_seq is None:
            cls_seq = np.arange(100).tolist()
        if len(cls_seq) != 1000:
            raise ValueError(f"ImageNet100 requires 1000 classes, but got {len(cls_seq)}")
        seq_100 = cls_seq[:100] 
        super().__init__(seq_100)

    def download_data(self, download_path="./data"):
        # assert download_path != "./data", "You should specify the folder of your ImageNet dataset"
        train_dir = f"{download_path}/train/"
        test_dir = f"{download_path}/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
