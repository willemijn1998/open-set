## PARTS BORROWED FROM: https://github.com/yue-zhongqi/gcm-cf 
### The Released Code for "Counterfactual Zero-shot and Open-Set Visual Recognition"
### Author: Wang Tan
### Part of Code borrow from "CGDL"

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN
from torch.utils.data import Dataset
import random

def get_dataset(data_name): 
    """
    This function takes the name of the required dataset and returns the train and test split. 
    """
    if  data_name == "MNIST": 
        transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize((0.1307,), (0.3081,))])
        trainset = MNIST(root='./data', train=True, transform=transform, download=True)
        testset = MNIST(root='./data', train=False, transform=transform, download=True)

    elif data_name == "CIFAR10": 
        trainset = CIFAR10(root='./data', train=True, download=True)
        testset = CIFAR10(root='./data', train=False, download=True)

    return trainset, testset 

def get_split(seed, n_classes, n_seen):
    "Sample the classes that will be seen/ unseen"
    random.seed(seed)
    seen_classes = random.sample(range(0, n_classes), n_seen)
    unseen_classes = [idx for idx in range(10) if idx not in seen_classes]
    return seen_classes, unseen_classes
    
class DatasetBuilder(Dataset): 
    def __init__(self, args): 
        self.trainset, self.testset = get_dataset(args.data_name)
        self.channel, self.height, self.width = self.trainset[0][0].shape
        self.n_classes = len(self.trainset.classes)
        self.n_seen = args.n_seen
        self.n_unseen = args.n_unseen
        self.seed = args.seed_sampler
        self.seen_classes, self.unseen_classes = get_split(self.seed, self.n_classes, self.n_seen)
        self.class2label = self.__class2label__()
        self.rename_classes()
        self.remove_unseen_train()
        self.val_train_split(args.val_ratio)

    def __class2label__(self): 
        class2label = {c: l for l, c in enumerate(self.seen_classes)}
        for c in self.unseen_classes: 
            class2label[c] = self.n_classes
        return class2label

    def rename_classes(self): 
        for item in self.class2label.items(): 
            self.trainset.targets[item[0]] = item[1]
            self.testset.targets[item[0]] = item[1]                      

    def remove_unseen_train(self):   
        "Remove the data from training set that is from an unseen class"  
        breakpoint()
        self.trainset.targets = self.trainset.targets[self.trainset.targets < self.n_seen]
        self.trainset.data = self.trainset.data[self.trainset.targets < self.n_seen]

    def val_train_split(self, val_ratio): 
        "Split the validation and train data"
        n_total = len(self.trainset.data)
        n_val = int(val_ratio * n_total)
        n_train = n_total - n_val
        self.trainset, self.valset = torch.utils.data.random_split(self.trainset, [n_train, n_val])