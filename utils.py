## Largely copied from: https://github.com/yue-zhongqi/gcm-cf 

### The Released Code for "Counterfactual Zero-shot and Open-Set Visual Recognition"
### Author: Wang Tan
### Part of Code borrow from "CGDL"

import numpy as np 
import torch 
import os
import shutil
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as T
import pandas as pd 

def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

def get_exp_name(args): 
	exp_name = []
	if args.epochs != 100 and args.data_name == 'CIFAR10': 
		exp_name.append('ep%s' %(args.epochs))
	if args.contra_loss: 
		exp_name.append('nu%s'%(args.nu))
		exp_name.append('T%s'%(args.temperature))
	if args.batch_size != 64: 
		exp_name.append('bs%s'%(args.batch_size))
	if args.mmd_loss: 
		exp_name.append('eta%s'%(args.eta))
	if args.supcon_loss: 
		exp_name.append('theta%s'%(args.theta))
	if args.debug: 
		exp_name = ['debug']+exp_name
	
	exp_name = '-'.join(exp_name)
			
	return exp_name

def get_transforms(s, p): 
	transform =  T.Compose([T.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s), 
							T.RandomGrayscale(p=p)])	

	return transform 

def sample_gaussian(m, v):
	# Remove the cuda() part since running on CPU
	# sample = torch.randn(m.shape).cuda()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	sample = torch.randn(m.shape).to(device)
	z = m + (v**0.5)*sample
	return z

def gaussian_parameters(h, dim=-1):
	m, h = torch.split(h, h.size(dim) // 2, dim=dim)
	v = F.softplus(h) + 1e-8
	return m, v

# def get_hparam_table(dict): 
# 	"outputs hparam table from dict"
# 	table = "Hyperparameters \n"
# 	for name, val in dict.items(): 
# 		table += '{}: {} \n'.format(name, val)
	
# 	return table 

def get_hparam_table(dict): 
	dfdict = {"Hyperparameters": list(dict.keys()), "Values": list(dict.values())}
	df = pd.DataFrame(dfdict)
	return df.to_markdown()


# def kl_normal(qm, qv, pm, pv):
# 	"""
# 	Args: 
# 		qm: y_latent_mu observed
# 		qv: y_latent_var observed 
# 		pm: prior y_latent_mu 
# 		pv: prior y_latent_var 
# 		yh: 
# 	returns: 
# 		KL divergence 

# 	"""
# 	element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
# 	kl = element_wise.sum(-1)
# 	#print("log var1", qv)
# 	return kl

def kl_normal(qm, qv, pm, pv, yh):
	element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm - yh).pow(2) / pv - 1)
	kl = element_wise.sum(-1)
	#print("log var1", qv)
	return kl

def get_mean_cov(W, targets, k, len_W, device): 
	means = torch.zeros((k, len_W)).to(device)
	covs = torch.zeros((k, len_W, len_W)).to(device)
	for i in range(k):
		means[i] = torch.mean(W[targets == i], dim=0)
		covs[i] = torch.cov((W[targets == i] - means[i]).T)
	
	return covs, means 

def mixup_data(x, y, alpha, device):
	"""Returns mixed inputs, pairs of targets, and lambda"""
	if alpha > 0:
		lam = np.random.beta(alpha, alpha)
	else:
		lam = 1

	batch_size = x.size()[0]
	index = torch.randperm(batch_size).to(device)

	mixed_x = lam * x + (1 - lam) * x[index, :]
	y_a, y_b = y, y[index]
	return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
	return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# def MMD_loss(W, W_aug, gamma): 
# 	"""
# 	Calculates the MMD loss between the W and the W_augmented from the same class. 
# 	args: 
# 		W: Class features of input images of shape N x D 
# 		W_aug: Class feature of augmented input images of shape M x D 
# 		gamma: Hyperparameter for gaussian kernel
# 	returns: 
# 		mmd_loss: Maximum Mean Discrepancy loss between distribution of regular and augmented images
# 	"""
# 	total = torch.cat((W, W_aug), dim=0)    
# 	N, M = W.shape[0], W_aug.shape[0] # N is batchsize W, M is batchsize W_augmented

# 	total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
# 	total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
# 	# total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
# 	# total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

# 	L2_distance = ((total0-total1)**2).sum(2) 

# 	kernels = torch.exp(gamma * L2_distance)
# 	XX = kernels[:N, :N].sum() / (N*(N-1))
# 	YY = kernels[N:, N:].sum() / (M*(M-1))
# 	XY = kernels[:N, N:].sum() / (N*M)
# 	YX = kernels[N:, :N].sum() / (M*N)

# 	mmd_loss = XX + YY - XY - YX
# 	return mmd_loss 

def MMD(x, y, kernel, device):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

	   Taken from: https://www.kaggle.com/code/onurtunali/maximum-mean-discrepancy/notebook

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
	
    return torch.mean(XX + YY - 2. * XY)


class LabelSmoothingLoss(nn.Module):
	def __init__(self, classes, smoothing=0.0, dim=-1):
		super(LabelSmoothingLoss, self).__init__()
		self.confidence = 1.0 - smoothing
		self.smoothing = smoothing
		self.cls = classes
		self.dim = dim

	def forward(self, pred, target):
		# pred = pred.log_softmax(dim=self.dim)
		with torch.no_grad():
			# true_dist = pred.data.clone()
			true_dist = torch.zeros_like(pred)
			true_dist.fill_(self.smoothing / (self.cls - 1))
			true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
		return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
