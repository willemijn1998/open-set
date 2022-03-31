import argparse 
import numpy as np
import torch
import torch.nn as nn
from utils import * 
from load_data import * 
from model import * 
import torch.utils.data as data
from train import * 
from evaluate import evaluate_model

parser = argparse.ArgumentParser(description='Open Set Recognition')

# dataset 
parser.add_argument('--data_name', type=str, default="MNIST", help='The dataset to use: MNIST, SVHN, CIFAR10, CIFAR100')
parser.add_argument('--n_unseen', type=int, default=4, help='Number of classes used as unseen.')
parser.add_argument('--n_seen', type=int, default=6, help='Number of classes used as seen.')
parser.add_argument('--data_path', type=str, default='./data', help='Which path to save data to.')
parser.add_argument('--val_ratio', type=float, default=0.1, help='Percentage of val set wrt train')

# optimization 
parser.add_argument('--epochs',  type=int, default=50, help='Number of epochs, default.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')

# model 
parser.add_argument('--len_Z', type=int, default=10, help='Length of latent feature Z')
parser.add_argument('--len_W', type=int, default=32, help='Length of latent feature W')
parser.add_argument('--beta_z', type=int, default=1, help='Hyperparam for KL div Z loss')
parser.add_argument('--lamda', type=int, default=100, help='Hyperparam for CE loss')
parser.add_argument('--nu', type=float, default=1.0, help='Hyperparam for contrastive loss')

# testing 
parser.add_argument('--eval', action='store_true', default=False, help='Only evaluate?')
parser.add_argument('--threshold_gauss', type=float, default=0.9, help='Threshold for Gaussian Model.')
# parser.add_argument('--threshold_recloss', type=float, default=2., help='How many standard devations for threshold')

# misc
parser.add_argument('--debug', action='store_true', default=False, help='If debug mode.')
parser.add_argument('--seed', type=str, default=42, help='The seed for reproducibility')
parser.add_argument('--seed_sampler', type=str, default=1234, help='The seed for sampling classes')

args = parser.parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if __name__ == '__main__': 

    latent_dim = args.len_Z + args.len_W
    exp_name = '%s_lamda%s_betaz%s_nu%s_lenZ%s_lenW%s' %(args.data_name, args.lamda, args.beta_z, args.nu, args.len_Z, args.len_W)
    
    if args.debug: 
        exp_name += '_debug'
        args.epochs = 1

    # build dataset 
    print("Preparing data...")
    dataset = DatasetBuilder(args)

    train_loader = data.DataLoader(dataset.trainset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = data.DataLoader(dataset.valset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = data.DataLoader(dataset.testset, batch_size=args.batch_szie, shuffle=False, drop_last=False)

    lvae = LVAE(in_ch=dataset.channel,
                out_ch64=64, out_ch128=128, out_ch256=256, out_ch512=512,
                kernel1=1, kernel2=2, kernel3=3, padding0=0, padding1=1, stride1=1, stride2=2,
                flat_dim32=32, flat_dim16=16, flat_dim8=8, flat_dim4=4, flat_dim2=2, flat_dim1=1,
                latent_dim512=512, latent_dim256=256, latent_dim128=128, latent_dim64=64, latent_dim32=latent_dim,
                num_class=args.num_classes, dataset=args.dataset, args=args)

    lvae.to(args.device)

    if args.eval: 
        # Load model and parameters
        model = ... 
        best_thresh = ... 
        best_feats = ... 

        # Test model 
        precision, recall, fscore = evaluate_model(model, test_loader, best_thresh, best_feats, args)

    else: 

        # train model 
        best_model, best_epoch, best_feats, best_loss, best_thresh = train(lvae, train_loader, val_loader, args)

        # Evaluate on test data 
        evaluate_model(best_model, test_loader, best_thresh, best_feats, args)

        # save model 
        # save unseen-seen split 
