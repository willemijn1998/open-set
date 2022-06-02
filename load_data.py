## PARTS BORROWED FROM: https://github.com/yue-zhongqi/gcm-cf 
### The Released Code for "Counterfactual Zero-shot and Open-Set Visual Recognition"
### Author: Wang Tan
### Part of Code borrow from "CGDL"

import torch
import torchvision
import torchvision.transforms as T
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN
import torch.utils.data as data
from torch.utils.data import Dataset
import random
from PIL import ImageFilter
import numpy as np
import cv2
from PIL import Image


def get_dataset(data_name, seed_sampler, n_classes, n_seen, seed): 
    """
    This function takes the name of the required dataset and returns the train and test split. 
    """
    channel = 3
    if  data_name == "MNIST": 
        mean = (0.1307,)
        std = (0.3081,)
        tform_train = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        tform_test = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

        trainset = MNIST(root='./data', train=True, transform=tform_train, download=True)
        testset = MNIST(root='./data', train=False, transform=tform_test, download=True)

        channel = 1

    elif data_name == "CIFAR10": 
        mean = [0.4914, 0.4822, 0.4465]
        std = (0.2023, 0.1994, 0.2010)
        tform = [T.ToTensor(), T.Normalize(mean, std)]

        tform_train = T.Compose([T.RandomResizedCrop(32, scale=(0.8, 1.0)),
                        T.RandomHorizontalFlip(0.5),
                        T.RandomRotation(10)] + tform )
        tform_test = T.Compose(tform)

        testset = CIFAR10(root='./data', train=False, transform=tform_test, download=True)
        trainset = CIFAR10(root='./data', train=True, transform=tform_train, download=True)
        testset.targets, trainset.targets = torch.LongTensor(testset.targets), torch.LongTensor(trainset.targets)

    elif data_name == "SVHN": 
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
        tform = [T.ToTensor(), T.Normalize(mean, std)]

        tform_train = T.Compose([T.RandomResizedCrop(32, scale=(0.7, 1.0))] + tform)
        tform_test = T.Compose(tform)

        testset = SVHN(root='./data', train=False, transform=tform_test, download=True)
        trainset = SVHN(root='./data', train=True, transform=tform_test, download=True)


    elif data_name == "CIFARaddN": 
        ... 
    
    seen_classes, unseen_classes = get_class_split(seed_sampler, n_classes, n_seen)
    trainset, valset, testset_seen, testset_unseen = construct_dataset(trainset, testset, seen_classes, unseen_classes, tform_train, tform_test, seed)

    return trainset, valset, testset_seen, testset_unseen, channel, seen_classes

def construct_dataset(trainset, testset, seen_classes, unseen_classes, tform_train, tform_test, seed):
    trainset = DatasetBuilder(
        [get_class_i(trainset.data, trainset.targets, idx) for idx in seen_classes],
        tform_train)

    testdata_seen = DatasetBuilder(
        [get_class_i(testset.data, testset.targets, idx) for idx in seen_classes],
        tform_test)

    testdata_unseen = DatasetBuilder(
        [get_class_i(testset.data, testset.targets, idx) for idx in unseen_classes],
        tform_test)
        
    valset, testset_seen, testset_unseen = val_test_split(testdata_seen, testdata_unseen, seed)

    return trainset, valset, testset_seen, testset_unseen


def val_test_split(testdata_seen, testdata_unseen, seed): 
    """ Creates val-test split from seen and unseen data"""

    n_val = int(len(testdata_seen)/2)
    n_test = len(testdata_seen) - n_val
    valset_seen, testset_seen = torch.utils.data.random_split(testdata_seen, [n_val, n_test], generator=torch.Generator().manual_seed(seed))

    n_val = int(len(testdata_unseen)/2)
    n_test = len(testdata_unseen) - n_val
    _, testset_unseen = torch.utils.data.random_split(testdata_unseen, [n_val, n_test], generator=torch.Generator().manual_seed(seed))

    return valset_seen, testset_seen, testset_unseen


def get_class_split(seed, n_classes, n_seen):
    "Sample the classes that will be seen/ unseen"
    random.seed(seed)
    seen_classes = random.sample(range(0, n_classes), n_seen)
    unseen_classes = [idx for idx in range(10) if idx not in seen_classes]
    return seen_classes, unseen_classes    

    
# class DatasetBuilder(Dataset): 
#     def __init__(self, data_name, n_seen, n_unseen, seed_sampler, val_ratio): 
#         self.trainset, self.testset, self.mean, self.std = get_dataset(data_name)
#         self.channel = 1 if data_name == 'MNIST' else 3
#         self.n_classes = len(self.trainset.classes)
#         self.n_seen = n_seen
#         self.n_unseen = n_unseen
#         self.seed = seed_sampler
#         self.seen_classes, self.unseen_classes = get_class_split(self.seed, self.n_classes, self.n_seen)
#         self.class2label = self.__class2label__()
#         self.rename_classes()
#         self.remove_unseen_train()
#         # Get split like yue21: 
#         breakpoint()
#         self.valset, self.testset = get_valtest_split(self.testset.data, self.testset.targets, self.n_seen)
#         # Get regular split: 

#     def __class2label__(self): 
#         """Creates a dictionary that maps the old class labels to the new ones, 
#         where all classes are labeled 0 - n_seen, and unseen classes as n_seen"""
#         class2label = {c: l for l, c in enumerate(self.seen_classes)}
#         for c in self.unseen_classes: 
#             class2label[c] = self.n_seen
#         return class2label

#     def rename_classes(self): 
#         new_train_targets = torch.zeros_like(self.trainset.targets)
#         new_test_targets = torch.zeros_like(self.testset.targets)
#         for item in self.class2label.items(): 
#             new_train_targets[self.trainset.targets == item[0]] = item[1]
#             new_test_targets[self.testset.targets == item[0]] = item[1]   
#         self.trainset.targets = new_train_targets
#         self.testset.targets = new_test_targets 

#     def remove_unseen_train(self):   
#         "Remove the data from training set that is from an unseen class"  
#         idx = (self.trainset.targets < self.n_seen).nonzero(as_tuple=True)[0]
#         self.trainset.targets = self.trainset.targets[idx]
#         self.trainset.data = self.trainset.data[idx]

#     def remove_unseen_val(self): 
#         "Remove the data from validation set that is from an unseen class"
#         idx  = 

#     def val_train_split(self): 
#         "Split the validation and train data"
        
                
#         n_total = len(self.trainset.data)
#         n_val = int(val_ratio * n_total)
#         n_train = n_total - n_val
#         self.trainset, self.valset = torch.utils.data.random_split(self.trainset, [n_train, n_val])

# --------------------------------------------------------------

def get_class_i(x, y, i):
    """
    x: trainset.train_data or testset.test_data
    y: trainset.train_labels or testset.test_labels
    i: class label, a number between 0 to 9
    return: x_i
    """
    # Convert to a numpy array
    y = np.array(y)
    # Locate position of labels that equal to i
    pos_i = np.argwhere(y == i)
    # Convert the result into a 1-D list
    pos_i = list(pos_i[:, 0])
    # Collect all data that match the desired label
    x_i = [x[j] for j in pos_i]
    return x_i

# borrow from https://gist.github.com/Miladiouss/6ba0876f0e2b65d0178be7274f61ad2f
class DatasetBuilder(Dataset):
    def __init__(self, datasets, transformFunc):
        """
        datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
        """
        self.datasets = datasets
        self.lengths = [len(d) for d in self.datasets]
        self.transformFunc = transformFunc

    def __getitem__(self, i):
        class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)

        img = self.datasets[class_label][index_wrt_class]

        if isinstance(img, torch.Tensor):
            img = Image.fromarray(img.numpy(), mode='L')
        elif type(img).__module__ == np.__name__:
            if np.argmin(img.shape) == 0:
                img = img.transpose(1, 2, 0)
            img = Image.fromarray(img)
        elif isinstance(img, tuple): #ImageNet
            # img = Image.open(img[0])
            img = cv2.imread(img[0])
            img = Image.fromarray(img)

        img = self.transformFunc(img)
        return img, class_label

    def __len__(self):
        return sum(self.lengths)

    def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
        """
        Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
        """
        # Which class/bin does i fall into?
        accum = np.add.accumulate(bin_sizes)
        if verbose:
            print("accum =", accum)
        bin_index = len(np.argwhere(accum <= absolute_index))
        if verbose:
            print("class_label =", bin_index)
        # Which element of the fallent class/bin does i correspond to?
        index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
        if verbose:
            print("index_wrt_class =", index_wrt_class)

        return bin_index, index_wrt_class