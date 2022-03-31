import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
# from torch.utils.tensorboard import SummaryWriter 
from copy import deepcopy


class DeterministicWarmup(object):
    """
    This class iterates through a linear increase of beta values from 0 to 1. 
    It is shown that annealing of beta works good. 
    """ 
    def __init__(self, n=100, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.inc = 1 / n

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc

        self.t = self.t_max if t > self.t_max else t  # 0->1
        return self.t

class ReconstructionLoss(object): 
    """ 
    This class incrementally updates the threshold for reconstruction
    losses. 
    """
    def __init__(self): 
        self.count = 0 
        self.sum = 0 
        self.sumSq = 0 
        
    def __add__(self, rec_losses): 
        self.count += rec_losses.shape[0]
        self.sum += rec_losses.sum()
        self.sumSq += (rec_losses * rec_losses).sum()
        
    def get_std(self): 
        var = (self.sumSq - (self.sum * self.sum)/ self.count) / (self.count-1)
        std = torch.sqrt(var).item()
        return std
    
    def get_mean(self): 
        return (self.sum / self.count).item()

    def get_threshold(self):
        return self.get_mean() + 2 * self.get_std()
        

class ClassFeatures(object): 
    """
    This class calculates the means and covariances of the class features W for each class Y
    in an incremental way. By adding more features, the means and cov are updated. 
    Input is the number of classes k and the length of the class features len_W. 
    """
    def __init__(self, k, len_W, seen_classes): 
        self.k = k
        self.sumW = torch.zeros((k, len_W))
        self.sumW2 = torch.zeros((k, len_W, len_W))
        self.count = torch.zeros((k, 1))

    def __add__(self, W, Y): 
        """
        Takes the correctly classified class feature W and its targets Y and updates 
        sum and sum squared
        """
        for i in range(self.k):
            self.sumW[i] += W[Y==i].sum(axis=0)
            self.sumW2[i] += torch.matmul(W[Y==i].T, W[Y==i])
            self.count[i] += W[Y==i].shape[0]

    def get_mean(self): 
        self.mean = self.sumW / self.count
        return self.mean
    
    def get_cov(self): 
        mean = self.get_mean()
        mean2 = torch.einsum('bi,bj->bij', mean, mean)
        N = self.count.view((self.k, 1, 1))
        self.cov = (self.sumW2 / N - mean2) * N / (N-1)
        return self.cov
                        


def train(model, train_loader, val_loader, args): 
    """
    This function trains the model given its arguments. 
    """

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 100, 150], gamma=0.1)
    beta = DeterministicWarmup(n=50, t_max=1) 
    best_val_loss = 1000

    for epoch in range(args.epochs): 
        # ---------------- TRAINING --------------------------
        print("Training... Epoch = %d" %epoch)
        
        model.train()
        true_preds, count = 0., 0
        classfeats = ClassFeatures(args.n_seen, args.len_W)
        rec_losses = ReconstructionLoss()

        
        for i, (input, labels) in enumerate(train_loader): 
            labels_onehot = F.one_hot(labels)

            input, labels, labels_onehot = input.to(args.device), labels.to(args.device), \
                                            labels_onehot.to(args.device)
            optimizer.zero_grad()

            # Y_pred is prediction from sampled W, Y_pred_mean is prediction from W mean           
            loss, W_mean, Y_pred, Y_pred_mean, x_re, rec, kl, ce = model.loss(input, labels, labels_onehot, next(beta), args.lamda, args)

            loss.backward()
            optimizer.step()

            true_preds += (Y_pred.argmax(dim=1) == labels).sum().item()
            count += labels.shape[0]

            # Add the correctly classified W_means and labels to the class features  
            classfeats.__add__(W_mean[(Y_pred.argmax(dim=1) == labels)], 
                                labels[(Y_pred.argmax(dim=1) == labels)])
            # Update the reconstruction loss threshold. 
            rec_loss = (x_re - input).pow(2).sum((3, 2, 1))
            rec_losses.__add__(rec_loss)

            if i % args.log_interval == 0: 
                print('[Run {}] Train Epoch: {} [{}/{} ({:.0f}%)]  lr:{}  loss:{:.3f} = rec:{:.3f} + kl:{:.3f} + ce:{:.3f}'.format(
                    args.run_idx, epoch, i * len(input), len(train_loader.dataset),
                           100. * i * len(input) / len(train_loader.dataset),
                           optimizer.param_groups[0]['lr'],
                           loss.data / (len(input)),
                           rec.data / (len(input)),
                           kl.data / (len(input)),
                           ce.data / (len(input))
                    ))

        model.eval() 
        # --------------- VALIDATION -----------------

        if epoch % args.val_interval == 0: 
            total_val_loss, true_preds_val, count = 0., 0, 0.

            for i, (input, labels) in enumerate(val_loader): 
                labels_onehot = F.one_hot(labels)
                input, labels, labels_onehot = input.to(args.device), labels.to(args.device), \
                                                labels_onehot.to(args.device)
                
                loss, W_mean, Y_pred, Y_pred_mean, x_re, rec, kl, ce = model.loss(input, labels, labels_onehot, next(beta), args.lamda, args)
                
                total_val_loss += loss
                true_preds_val += (Y_pred.argmax(dim=1) == labels).sum().item()
                count += labels.shape[0]

            val_loss = total_val_loss / count 
            val_acc = true_preds_val / count 
            print('===> Epoch: {} Val loss: {}, Val Acc: {}'.format(epoch, val_loss, val_acc))

            if val_loss < best_val_loss:
                best_model = deepcopy(model)
                best_epoch = epoch 
                best_feats = deepcopy(classfeats)
                best_rec_losses = deepcopy(rec_losses)

                val_loss = best_val_loss 

                print('New best val loss!')

        scheduler.step() 

    best_thresh = best_rec_losses.get_threshold()

    return best_model, best_epoch, best_feats, best_val_loss, best_thresh



