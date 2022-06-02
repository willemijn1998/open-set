import argparse
from calendar import c 
import numpy as np
import torch
import torch.nn as nn
from utils import * 
from load_data import * 
from model import * 
import torch.utils.data as data
from train import * 
from evaluate import evaluate_model, select_threshold
from evaluate import get_probs
import time
import sys 
import pandas as pd 

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
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--momentum', type=float, default=0.01, help='Momentum for optimizer')
parser.add_argument('--wd', type=float, default=0.0, help='Weight decay for optimization')
parser.add_argument('--lr_decay', type=float, default=0.1, help='Weight decay for learning rate')
parser.add_argument('--milestones', type=int, default=[60, 100, 150], help='Milestones (epochs) after which lr decay')

# model 
parser.add_argument('--len_Z', type=int, default=10, help='Length of latent feature Z')
parser.add_argument('--len_W', type=int, default=32, help='Length of latent feature W')
parser.add_argument('--beta_z', type=int, default=1, help='Hyperparam for KL div Z loss')
parser.add_argument('--lamda', type=int, default=100, help='Hyperparam for CE loss')
parser.add_argument('--nu', type=float, default=1.0, help='Hyperparam for contrastive loss')
parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for scaling contrastive loss')


# testing 
parser.add_argument('--eval', action='store_true', default=False, help='Only evaluate?')
parser.add_argument('--threshold_ood', type=float, default=25, help='Threshold for Gaussian Model.')

# misc
parser.add_argument('--debug', action='store_true', default=False, help='If debug mode.')
parser.add_argument('--seed', type=str, default=1, help='The seed for reproducibility')
parser.add_argument('--seed_sampler', type=str, default=1234, help='The seed for sampling classes')
parser.add_argument('--log_interval', type=int, default=20, help='After how many epochs logging training statistics.')
parser.add_argument('--val_interval', type=int, default=5, help='How often validate on val set')

args = parser.parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__': 

    latent_dim = args.len_Z + args.len_W

    if not os.path.isdir('./results/%s/sample%s'%(args.data_name, args.seed_sampler)): 
        os.makedirs('./results/%s/sample%s'%(args.data_name, args.seed_sampler))

    args.exp_name = '%s/sample%s/seed%s_lam%s_z%s_nu%s_Z%s_W%s' %(args.data_name, args.seed_sampler, args.seed, args.lamda, args.beta_z, args.nu, args.len_Z, args.len_W)
    print('Running experiment {}'.format(args.exp_name))
    args.n_classes = args.n_seen + args.n_unseen
    if args.debug: 
        args.exp_name += '_debug'
        args.epochs = 1

    # build dataset 
    print("Preparing data...")
    dataset = DatasetBuilder(args.data_name, args.n_seen, args.n_unseen, args.seed_sampler, args.val_ratio)

    train_loader = data.DataLoader(dataset.trainset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = data.DataLoader(dataset.valset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = data.DataLoader(dataset.testset, batch_size=args.batch_size, shuffle=False, drop_last=False)
   
    # Load model and results 
    model = LVAE(in_ch=dataset.channel, latent_dim32=latent_dim, num_class=args.n_seen, dataset=args.data_name, args=args)
    model.to(args.device)
    results = torch.load('./results/{}_model'.format(args.exp_name), map_location=args.device)
    model.load_state_dict(results['state_dict'])   

    best_ood, best_threshold = select_threshold(model, results['class_means'], results['class_covs'], test_loader, args)

    print('best threshold for {} is {} with ood score {}'.format(args.data_name, best_threshold, best_ood))
    args.threshold_ood = best_threshold

    # Evaluate model 
    print('Evaluating on test set')
    model.to(args.device)
    precision, recall, fscore, acc, predictions, targets = evaluate_model(model, test_loader, results['rec_thresh'], \
                                                args.threshold_ood, results['class_means'], results['class_covs'], args)


    print("Precision: {} Recall: {} F-score: {} Acc: {}".format(precision, recall, fscore, acc))
    performance = {'Precision': precision, 'Recall' : recall, 'F-Score': fscore}
    performance_df = pd.DataFrame.from_dict(performance)
    performance_df.to_csv('results/{}_{}_performance.csv'.format(args.exp_name, args.threshold_ood))
    
    with open('results/{}_{}_predictions'.format(args.exp_name, args.threshold_ood), 'wb') as f: 
        np.save(f, predictions)

    with open('results/{}_{}_targets'.format(args.exp_name, args.threshold_ood), 'wb') as f: 
        np.save(f, targets)

    
