import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
# from torch.utils.tensorboard import SummaryWriter 
from copy import deepcopy
# from model import *
from modelo import *
from utils import * 


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
    def __init__(self, k, len_W, device): 
        self.k = k
        self.sumW = torch.zeros((k, len_W)).to(device)
        self.sumW2 = torch.zeros((k, len_W, len_W)).to(device)
        self.count = torch.zeros((k, 1)).to(device)

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


def train(model, train_loader, val_loader, n_seen, writer, args): 
    """
    This function trains the model given its arguments. 
    """
    
    # writer = SummaryWriter('runs/{}'.format(args.exp_name))

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_decay)
    beta = DeterministicWarmup(n=50, t_max=1) 
    # scaler = torch.cuda.amp.GradScaler()
    best_val_loss = 1000
    model.supcon_critic = SupConLoss(args.con_temperature)

    for epoch in range(args.epochs): 
        # ---------------- TRAINING --------------------------
        print("Training... Epoch = %d" %epoch)

        correct_Wmean = []
        correct_targets = []
        
        model.train()
        true_preds, count = 0., 0
        # classfeats = ClassFeatures(args.n_seen, args.len_W, args.device)
        rec_losses = ReconstructionLoss()

        if epoch in args.decreasing_lr:
            optimizer.param_groups[0]['lr'] *= args.lr_decay
            print("~~~learning rate:", optimizer.param_groups[0]['lr'])
        
        for i, (input, labels) in enumerate(train_loader): 
            if args.debug and i == 3: 
                break
            
            input, labels = input.to(args.device), labels.to(args.device)
            labels_onehot = F.one_hot(labels, num_classes = n_seen).type(torch.float)


            input, labels, labels_onehot = input.to(args.device), labels.to(args.device), \
                                            labels_onehot.to(args.device)

            # with torch.cuda.amp.autocast(): 
            loss, W_mean, Y_pred, Y_pred_mean, x_re, rec, kl, ce = model.loss(input, labels, labels_onehot, next(beta), args.lamda, args)
            
            # writer.add_embedding(W_mean, metadata=labels, tag='W_mean', global_step=epoch)
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            # optimizer.zero_grad()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            true_preds += (Y_pred.argmax(dim=1) == labels).sum().item()
            count += labels.shape[0]

            # Add the correctly classified W_means and labels to the class features  
            # classfeats.__add__(W_mean[(Y_pred.argmax(dim=1) == labels)], 
            #                     labels[(Y_pred.argmax(dim=1) == labels)])

            # ADD: with torch.no_grad(): 
            cor_fea = W_mean[(Y_pred.argmax(dim=1) == labels)]
            cor_tar = labels[(Y_pred.argmax(dim=1) == labels)]
            correct_Wmean.append(cor_fea)
            correct_targets.append(cor_tar)

            # print("Batch {}: loss: {} rec: {} kl: {} ce:{} ".format(i, loss, rec, kl, ce))
            # print("targets: {}".format(labels))
            # print("Class feats, shape: {} \n cor_fea[0]: {}".format(cor_fea.shape, cor_fea[0]))

            # Update the reconstruction loss threshold. 
            rec_loss = (x_re - input).pow(2).sum((3, 2, 1))
            rec_losses.__add__(rec_loss)

            if i % args.log_interval == 0: 
                print('Train Epoch: {} [{}/{} ({:.0f}%)]  lr:{}  loss:{:.3f} = rec:{:.3f} + kl:{:.3f} + ce:{:.3f}'.format(
                    epoch, i * len(input), len(train_loader.dataset),
                           100. * i * len(input) / len(train_loader.dataset),
                           optimizer.param_groups[0]['lr'],
                           loss.data / (len(input)),
                           rec.data / (len(input)),
                           kl.data / (len(input)),
                           ce.data / (len(input))
                    ))
                    
        train_acc = float(100 * true_preds) / len(train_loader.dataset)   
        writer.add_scalar("Loss/train", loss.data/len(input), epoch)
        writer.add_scalar("Reconstruction/train", rec.data/ len(input), epoch)
        writer.add_scalar("KL/train", kl.data/len(input), epoch)
        writer.add_scalar("CE/train", ce.data/len(input), epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        if args.contra_loss: 
            writer.add_scalar("Contrastive/train", model.contra_loss.data / len(input), epoch)
        if args.mmd_loss: 
            writer.add_scalar("MMD/train", model.invar_loss.data / len(input), epoch)
        if args.supcon_loss: 
            writer.add_scalar("SupCon/train", model.supcon_loss.data/len(input), epoch)


        model.eval() 
        # --------------- VALIDATION -----------------

        if epoch % args.val_interval == 0: 
            print('Validating... ')
            total_loss, total_true_preds, total_kl, total_rec, total_ce = 0., 0, 0., 0., 0.
            
            with torch.no_grad(): 
                for i, (input, labels) in enumerate(val_loader): 
                    # Only validate on 
                    if args.debug and i == 3: 
                        break
                    labels_onehot = F.one_hot(labels, num_classes = n_seen).type(torch.float)
                    input, labels, labels_onehot = input.to(args.device), labels.to(args.device), \
                                                    labels_onehot.to(args.device)
                    
                    loss, W_mean, Y_pred, Y_pred_mean, x_re, rec, kl, ce = model.loss(input, labels, labels_onehot, next(beta), args.lamda, args)
                    
                    total_loss += loss
                    total_true_preds += (Y_pred.argmax(dim=1) == labels).sum().item()
                    total_kl += kl
                    total_rec += rec
                    total_ce += ce 

                n = len(val_loader.dataset)
                val_loss = total_loss/ n
                val_acc = total_true_preds/n

                print('===> Epoch: {} Val loss: {}, Val Acc: {}'.format(epoch, val_loss, val_acc))
                writer.add_scalar("Loss/val", val_loss, epoch)
                writer.add_scalar("Reconstruction/val", total_rec/n, epoch)
                writer.add_scalar("KL/val", total_kl/n, epoch)
                writer.add_scalar("CE/val", total_ce/n, epoch)
                writer.add_scalar("Accuracy/val", val_acc, epoch)
                if args.contra_loss: 
                    writer.add_scalar("Contrastive/val", model.contra_loss.data / n, epoch)
                if args.mmd_loss: 
                    writer.add_scalar("MMD/val", model.invar_loss.data / n, epoch)
                if args.supcon_loss: 
                    writer.add_scalar("SupCon/val", model.supcon_loss.data/len(input), epoch)


                if val_loss < best_val_loss: 
                    best_state_dict = deepcopy(model.state_dict())

                    W = torch.cat(correct_Wmean, axis=0)
                    targets = torch.cat(correct_targets)
                    covs, means = get_mean_cov(W, targets, args.n_seen, args.len_W, args.device)
                    # print('Covs: {} \n Means: {}'.format(covs, means))

                    # class_means = classfeats.get_mean()
                    # class_covs = classfeats.get_cov()
                    # best_corr_wmean = correct_Wmean
                    # best_corr_targets = correct_targets

                    best_rec_thresh = rec_losses.get_threshold()
                    best_epoch = epoch 
                    best_val_loss = val_loss 
 
                    print('New best val loss!: {}'.format(best_val_loss))

        # scheduler.step() 

    print('End of training. Best epoch: {}, Best val loss: {}'.format(best_epoch, best_val_loss))
    # W = torch.cat(best_corr_wmean, axis=0)
    # targets = torch.cat(best_corr_targets)
    # covs, means = get_mean_cov(W, targets, args.n_seen, args.len_W, args.device)
    
    return best_state_dict, means, covs, best_rec_thresh, best_val_loss

