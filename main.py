import argparse 
import numpy as np
import torch
import torch.nn as nn
from utils import * 
from load_data import * 
# from model import * 
from modelo import *
import torch.utils.data as data
from train import * 
import time
import sys 
import pandas as pd 
from torch.utils.tensorboard import SummaryWriter 
from evaluate import *
from datetime import datetime


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

parser = argparse.ArgumentParser(description='Open Set Recognition')

# dataset 
parser.add_argument('--data_name', type=str, default="MNIST", help='The dataset to use: MNIST, SVHN, CIFAR10, CIFAR100')
parser.add_argument('--n_unseen', type=int, default=4, help='Number of classes used as unseen.')
parser.add_argument('--n_seen', type=int, default=6, help='Number of classes used as seen.')
parser.add_argument('--data_path', type=str, default='./data', help='Which path to save data to.')

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
parser.add_argument('--nu', type=int, default=10, help='Hyperparam for contrastive loss')
parser.add_argument('--temperature', type=int, default=10, help='Temperature for scaling contrastive loss')
parser.add_argument('--ood_bound', type=float, default=0.21, help='fraction of in distr data to be regarded as ood')
parser.add_argument('--mmd_loss', action='store_true', default=False, help='Use cf_invar function')
parser.add_argument('--s_jitter', type=float, default=0.5, help='Strength for color jitter')
parser.add_argument('--p_grayscale', type=float, default=0.3, help='Probability of random grayscale')
parser.add_argument('--gamma', type=float, default=1.0, help='Hparam for kernel MMD loss')
parser.add_argument('--eta', type=int, default=1, help='Hparam for mmd loss')
parser.add_argument('--kernel', type=str, default='multiscale', help='Which kernel to use for MMD loss')
parser.add_argument('--contra_loss', action='store_true', default=False, help='Use contra_loss function')
parser.add_argument('--con_temperature', type=float, default=0.07, help='Temperature for supcon loss')
parser.add_argument('--theta', type=float, default=0.2, help='Hparam for supervised contra loss')

# testing 
parser.add_argument('--eval', action='store_true', default=False, help='Only evaluate?')
parser.add_argument('--threshold_ood', type=float, default=None, help='Threshold for Gaussian Model.')
parser.add_argument('--eval_id', action='store_true', default=False, help='Eval only on in distribution data?')
parser.add_argument('--threshold_yue', type=float, default=0.9, help='Threshold for gaussian model Yue')

# misc
parser.add_argument('--debug', action='store_true', default=False, help='If debug mode.')
parser.add_argument('--seed', type=int, default=117, help='The seed for reproducibility')
parser.add_argument('--seed_sampler', type=str, default=777, help='The seed for sampling classes')
parser.add_argument('--log_interval', type=int, default=20, help='After how many epochs logging training statistics.')
parser.add_argument('--val_interval', type=int, default=5, help='How often validate on val set')
parser.add_argument('--check_probs', action='store_true', default=False, help='check unseen probs')
parser.add_argument('--decreasing_lr', default='60,100,150', help='decreasing strategy')
parser.add_argument('--yue', action='store_true', default=False, help='Use Yue ood method')
parser.add_argument('--supcon_loss', action='store_true', default=False, help='Whether to use supervised contra loss')
parser.add_argument('--mutual_info', action='store_true', default=False, help='Use Mutual Info loss term?')
parser.add_argument('--mmd_logits_loss', action='store_true', default=False, help='Use MMD loss on logits?')
parser.add_argument('--no_rec_loss', action='store_true', default=False, help='NOT use reconstruction loss for ood')

if __name__ == '__main__': 

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    set_seed(args.seed)
    args.time = datetime.now()    
    args.decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

    args.n_classes = args.n_seen + args.n_unseen

    if args.mmd_loss: 
        args.transforms = get_transforms(args.s_jitter, args.p_grayscale)    

    args.exp_name = get_exp_name(args)
    args.save_path = '%s/%s/sample%s/seed%s'%(args.data_name, args.exp_name, args.seed_sampler, args.seed)

    if not os.path.isdir('./results/' + args.save_path): 
        os.makedirs('./results/' + args.save_path) 

    args_dict = vars(args)
    config = vars(args)
    hparam_table = get_hparam_table(args_dict)

    print('Running experiment {} with hyperparameters: {}'.format(args.exp_name, args_dict))

    writer = SummaryWriter("runs/%s"%(args.save_path))
    writer.add_text("Hyperparameters", hparam_table)

    # wandb.init(projec="CIFAR10-")
    
    # ------- BUILD DATASET --------------
    print("Preparing data...")
    trainset, valset, testset_seen, testset_unseen, channel, seen_classes = get_dataset(args.data_name, args.seed_sampler, args.n_classes, args.n_seen, args.seed)

    train_loader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=1, pin_memory=True, generator=torch.Generator().manual_seed(args.seed), worker_init_fn=seed_worker)
    val_loader = data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=1, pin_memory=True, generator=torch.Generator().manual_seed(args.seed), worker_init_fn=seed_worker)
    test_loader_seen = data.DataLoader(testset_seen, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=1, pin_memory=True, generator=torch.Generator().manual_seed(args.seed), worker_init_fn=seed_worker)
    test_loader_unseen = data.DataLoader(testset_unseen, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=1, pin_memory=True, generator=torch.Generator().manual_seed(args.seed), worker_init_fn=seed_worker)

    latent_dim = args.len_Z + args.len_W

    model = LVAE(in_ch=channel, latent_dim32=latent_dim, num_class=args.n_seen, dataset=args.data_name, args=args)

    model.to(args.device)    

    if not args.eval: 
    # ------- TRAIN MODEL --------------


        state_dict, class_means, class_covs, rec_thresh, val_loss = train(model, train_loader, val_loader, args.n_seen, writer, args)

        torch.save({
                        'state_dict': state_dict, 
                        'class_means': class_means, 
                        'class_covs': class_covs, 
                        'rec_thresh': rec_thresh, 
                        'val_loss': val_loss, 
                        'seen_classes': seen_classes, 
                        'args': args_dict
                    }, './results/{}/model.pkl'.format(args.save_path))
        
    results = torch.load('./results/{}/model.pkl'.format(args.save_path), map_location=args.device)
    model.load_state_dict(results['state_dict'])    
    
    # ------- EVALUATE MODEL --------------
    model.eval()
    if args.eval_id: 
        args.means = results['class_means']
        args.covs = results['class_covs']
        precision, recall, fscore, acc = eval_in_distribution(model, test_loader_seen, args)
        print("Precision: {} Recall: {} F-score: {} Acc: {} ".format(precision, recall, fscore, acc))
    
    if not args.threshold_ood:
        args.threshold_ood = select_threshold(model, results['class_means'], results['class_covs'], val_loader, train_loader, args.ood_bound, args, test_loader_seen, test_loader_unseen)


    print('Evaluating on test set')
    if args.no_rec_loss:
        rec_threshold = 1000000
    else: 
        rec_threshold = results['rec_thresh']

    if args.yue: 
        precision, recall, fscore, acc = evaluate_model_yue(model, test_loader_seen, test_loader_unseen, rec_threshold, args.threshold, results['class_means'], results['class_covs'], args)


    precision, recall, fscore, acc, auc = evaluate_model(model, test_loader_seen, test_loader_unseen, rec_threshold, \
                                                args.threshold_ood, results['class_means'], results['class_covs'], args)



    f1_average = np.mean(fscore)
    print("With threshold {}: \n Precision: {} Recall: {} F-score: {} Acc: {} F1 av:{} AUC: {}".format(args.threshold_ood, precision, recall, fscore, acc, f1_average, auc))
    threshold = torch.cat((args.threshold_ood, torch.zeros(1)))
    performance = {'Precision': precision, 'Recall' : recall, 'F-Score': fscore, 'Threshold': threshold}
    performance_df = pd.DataFrame.from_dict(performance)
    performance_df.to_csv('results/{}/performance.csv'.format(args.save_path))
    

    
