import torch 
import numpy as np
# if not torch.cuda.is_available(): 
# from rpy2.robjects.packages import importr
# from rpy2.robjects import numpy2ri
# numpy2ri.activate()
# mvt = importr('mvtnorm')
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, f1_score, roc_auc_score
from scipy.stats import multivariate_normal
import torch.distributions.multivariate_normal as mvn

def eval_in_distribution(model, loader_seen, args): 
    true_preds = 0
    N = 0
    preds_list = []
    target_list = []

    with torch.no_grad(): 
        for input, labels in loader_seen: 
            labels_onehot = F.one_hot(labels, num_classes = args.n_seen).type(torch.float)
            input, labels, labels_onehot = input.to(args.device), labels.to(args.device), \
                                                labels_onehot.to(args.device)

            mu_test, output_test, x_re = model.test(input, labels_onehot, args)
            true_preds += (output_test.argmax(dim=1) == labels).sum().item()
            N += len(labels)
            
            pred = output_test.argmax(dim=1)
            preds_list.append(pred)
            target_list.append(labels)
    
    acc = true_preds / N
    targets = torch.cat(target_list, 0)
    predictions = torch.cat(preds_list, 0)

    (precision, recall, fscore, _) = precision_recall_fscore_support(targets.cpu(), predictions.cpu(), average=None, labels=np.arange(args.n_seen+1))


    return precision, recall, fscore, acc


def evaluate_model(model, loader_seen, loader_unseen, rec_threshold, ood_threshold, means, covs, args): 
    """ 
    This function evaluates the model with given rec threshold and class features.
    Args: 
        model: model to evaluate 
        loader_seen: data loader to evaluate on (test, validation, train)
        loader_unseen: 
        rec_threshold: learned reconstruction error threshold for OSR 
        ood_treshold: shape args.n_seen
        means: 
        covs: 
        args: other arguments 
    Returns: 
        precision: 
        recall: 
        fscore: 

    """
    true_preds = 0.
    preds_list = []
    target_list = []
    prob_list = []
    recloss_list = []
    N = 0 
    with torch.no_grad(): 
        for input, labels in loader_unseen: 
            labels = torch.ones(labels.shape) * args.n_seen
            labels_onehot = torch.zeros((len(labels), args.n_seen)).to(args.device)
            input, labels = input.to(args.device), labels.to(args.device)

            W, output, x_rec = model.test(input, labels_onehot, args)

            pred = output.argmax(dim=1)

            # If reconstruction loss is above threshold: predict unseen 
            rec_loss = (x_rec - input).pow(2).sum((3, 2, 1))
            pred[(rec_loss > rec_threshold)] = args.n_seen


            # If gaussian distance above threshold: predict unseen
            probs = torch.zeros(len(labels), args.n_seen)
            for i in range(args.n_seen): 

                m = mvn.MultivariateNormal(loc=means[i], covariance_matrix=covs[i])
                prob = m.log_prob(W)
                probs[:,i] = prob
            pred[torch.all((probs-ood_threshold) < 0, 1)] = args.n_seen
            
            recloss_list.append(rec_loss)
            prob_list.append(probs)
            preds_list.append(pred)
            target_list.append(labels)
            true_preds += (pred == labels).sum().item()
            N += len(labels)

        for input, labels in loader_seen: 
            labels_onehot = F.one_hot(labels, num_classes = args.n_seen).type(torch.float).to(args.device)
            input, labels = input.to(args.device), labels.to(args.device)

            W, output, x_rec = model.test(input, labels_onehot, args)

            pred = output.argmax(dim=1)

            # If reconstruction loss is above threshold: predict unseen 
            rec_loss = (x_rec - input).pow(2).sum((3, 2, 1))
            pred[(rec_loss > rec_threshold)] = args.n_seen

            # If gaussian distance above threshold: predict unseen
            probs = torch.zeros(len(labels), args.n_seen)
            for i in range(args.n_seen): 
                m = mvn.MultivariateNormal(loc=means[i], covariance_matrix=covs[i])
                prob = m.log_prob(W)
                probs[:,i] = prob
            pred[torch.all((probs-ood_threshold) < 0, 1)] = args.n_seen

            recloss_list.append(rec_loss)
            prob_list.append(probs)
            preds_list.append(pred)
            target_list.append(labels)
            true_preds += (pred == labels).sum().item()
            N+=len(labels)
    
    acc = true_preds / N
    targets = torch.cat(target_list, 0)
    probabilities = torch.cat(prob_list, 0)
    rec_losses = torch.cat(recloss_list)
    max_probs, _ = torch.max(probabilities, 1)
    binary_targets = torch.zeros(targets.shape)
    binary_targets[targets == args.n_seen] = 1
    predictions = torch.cat(preds_list, 0)
    breakpoint()

    (precision, recall, fscore, support) = precision_recall_fscore_support(targets.cpu(), predictions.cpu(), average=None, labels=np.arange(args.n_seen+1))
    auc_score = roc_auc_score(binary_targets.cpu(), max_probs.cpu())

    return precision, recall, fscore, acc, auc_score

def multivariateGaussian(vector, mu, sigma):
    """
    Args: 
        vector: testfea[i], test feature output of Y, shape (32,)
        mu: mu[i], calculated means per class, shape (32,)
        sigma: sigma[i], calculated cov matrix per class, shape (32,32)
    Returns:
        probability of vector in distribution 

    """
    vector = np.array(vector)
    dim=len(mu)
    # Mahalanobis distance
    # (32,) * (32,32) * (32,) = (32,32) * (32,) = (32,32)
    d = (np.mat(vector-mu)) * np.mat(np.linalg.pinv(sigma))*(np.mat(vector - mu).T)
    # probability density of point 'vector' 
    p = np.exp(-d/2) /(((2*np.pi) ** (dim/2)) * (np.linalg.det(sigma)) ** (0.5))
    p = float(p)
    return p

# def multivariateGaussian(vector, mu, sigma): 
#     delta = (vector - mu) 
#     d = torch.dot(delta, torch.matmul(torch.inverse(sigma), delta))
#     p = torch.exp(-d/2) / ((2*torch.pi)**(dim/2)) * (torch.linalg.det(sigma) ** 0.5)


def multivariateGaussianNsigma(sigma,threshold):
    # get quantile x for which -x <= X <= x
    # sigma is cov matrix, threshold = p = 0.5
    q = np.array(mvt.qmvnorm(threshold, sigma = sigma, tail = "both")[0]) 
    n = q[0]
    dim= sigma.shape[0]
    m = (np.diag(sigma) ** 0.5) * n
    d = (np.mat(m) * np.mat(np.linalg.pinv(sigma)) * (np.mat(m).T)) 
    p = np.exp(d * (-0.5)) / (((2 * np.pi) ** int(dim/2)) * (np.linalg.det(sigma)) ** (0.5))
    return p

def evaluate_model_yue(model, loader_seen, loader_unseen, rec_threshold, qmv_threshold, means, covs, args): 
    """ 
    This function evaluates the model with given rec threshold and class features.
    Args: 
        model: model to evaluate 
        loader_seen: data loader to evaluate on (test, validation, train)
        loader_unseen: 
        rec_threshold: learned reconstruction error threshold for OSR 
        ood_treshold: shape args.n_seen
        means: 
        covs: 
        args: other arguments 
    Returns: 
        precision: 
        recall: 
        fscore: 
    """
    true_preds = 0.
    preds_list = []
    target_list = []
    pNsigma = torch.zeros(args.n_seen)
    for i in range(args.n_seen): 
        pNsigma[i] = multivariateGaussianNsigma(covs[i].detach().numpy(),qmv_threshold)

    with torch.no_grad(): 
        for input, labels in loader_unseen: 
            labels = torch.ones(labels.shape) * args.n_seen
            labels_onehot = torch.zeros((len(labels), args.n_seen)).to(args.device)
            input, labels = input.to(args.device), labels.to(args.device)

            W, output, x_rec = model.test(input, labels_onehot, args)

            pred = output.argmax(dim=1)

            # If reconstruction loss is above threshold: predict unseen 
            rec_loss = (x_rec - input).pow(2).sum((3, 2, 1))
            pred[(rec_loss > rec_threshold)] = args.n_seen

            # If gaussian distance above threshold: predict unseen

            n = len(labels)
            probs = torch.zeros(n, args.n_seen)
            for i in range(args.n_seen): 
                for j in range(n): 
                    probs[i,j] = multivariateGaussian(W[j].numpy(), means[i].numpy(), covs[i].numpy())
            
            pred[torch.all((probs-pNsigma) < 0, 1)] = args.n_seen

            preds_list.append(pred)
            target_list.append(labels)
            true_preds += (pred == labels).sum().item()

        for input, labels in loader_seen: 
            labels_onehot = F.one_hot(labels, num_classes = args.n_seen).type(torch.float).to(args.device)
            input, labels = input.to(args.device), labels.to(args.device)

            W, output, x_rec = model.test(input, labels_onehot, args)

            pred = output.argmax(dim=1)

            # If reconstruction loss is above threshold: predict unseen 
            rec_loss = (x_rec - input).pow(2).sum((3, 2, 1))
            pred[(rec_loss > rec_threshold)] = args.n_seen

            # If gaussian distance above threshold: predict unseen
            n = len(labels)
            probs = torch.zeros(n, args.n_seen)
            for i in range(args.n_seen): 
                for j in range(n): 
                    probs[i,j] = multivariateGaussian(W[j].numpy(), means[i].numpy(), covs[i].numpy())
                    
            
            pred[torch.all((probs-pNsigma) < 0, 1)] = args.n_seen

            preds_list.append(pred)
            target_list.append(labels)
            true_preds += (pred == labels).sum().item()


    acc = true_preds / (len(loader_seen) + len(loader_unseen)) 
    targets = torch.cat(target_list, 0)
    predictions = torch.cat(preds_list, 0)

    (precision, recall, fscore, support) = precision_recall_fscore_support(targets.cpu(), predictions.cpu(), average=None, labels=np.arange(args.n_seen+1))

    return precision, recall, fscore, acc


def get_probs(model, means, covs, loader, args): 
    "This function returns the probability of all data points in loader per class: n x k"
    prob_list = []
    labels_list = []
    preds_list = []
    with torch.no_grad(): 
        for input, labels in loader: 
            input, labels  = input.to(args.device), labels.to(args.device)
            W, output, _ = model.test(input, args)
            preds = output.argmax(dim=1)
            probs = torch.zeros((input.shape[0], args.n_seen))

            for i in range(args.n_seen): 
                m = mvn.MultivariateNormal(loc=means[i], covariance_matrix=covs[i])
                prob = m.log_prob(W)
                probs[:,i] = prob
            prob_list.append(probs)
            labels_list.append(labels)
            preds_list.append(preds)
        
    all_probs = torch.cat(prob_list, 0)
    all_labels = torch.cat(labels_list)
    all_preds = torch.cat(preds_list)

    return all_probs, all_labels, all_preds      

def get_class_features(model, loader, args): 
    """
    Get all class features from the validation set 
    """
    all_W = []
    all_labels = []

    with torch.no_grad(): 
        for input, labels in loader: 
            input, labels  = input.to(args.device), labels.to(args.device)
            labels_onehot = F.one_hot(labels, num_classes = args.n_seen).type(torch.float)

            W, _, _ = model.test(input, labels_onehot, args)
            all_W.append(W)
            all_labels.append(labels)
    
    all_W = torch.cat(all_W, 0)
    all_labels = torch.cat(all_labels)

    return all_W, all_labels 

def get_probs_per_class(class_feats, labels, n_seen, means, covs): 
    """
    Get the probability of data points belonging to its label 
    """

    probs_per_class = {}
    with torch.no_grad(): 
        for i in range(n_seen): 
            m = mvn.MultivariateNormal(loc=means[i], covariance_matrix=covs[i])
            probs = m.log_prob(class_feats[labels==i])
            probs_per_class[i] = probs

    return probs_per_class

def select_threshold(model, means, covs, val_loader, train_loader, lower_bound, args, test_loader_seen, test_loader_unseen): 
    """
    This function selects the best threshold per class. 
    args: 
        model
        means: class features means as calculated by model on training data 
        covs: class feature covariance matrix from training data 
        loader: validation loader for hparam selection 
        lowerbound: percentage of in-distribution data to fall out of distribution
    returns: 
        threshold per class: torch.Tensor shape args.n_seen where each dimension 
            corresponds to the threshold of that class, e.g. torch.Tensor([thres0,thresh1,...])
    """
    class_feats, labels = get_class_features(model, val_loader, args)
    # class_feats_train, labels_train = get_class_features(model, train_loader, args)
    # class_feats = torch.cat((class_feats_val, class_feats_train), 0)
    # labels = torch.cat((labels_val, labels_train))
    probs_per_class = get_probs_per_class(class_feats, labels, args.n_seen, means, covs)
    threshold = torch.zeros(args.n_seen)
    for label, probs in probs_per_class.items(): 
        threshold[label] = torch.quantile(probs, lower_bound)

    return threshold 


# def select_threshold(model, means, covs, loader, args):
#     """This function calculates optimal threshold epsilon"""
#     threshold = torch.zeros(args.n_seen) 
#     all_probs, all_labels, all_preds = get_probs(model, means, covs, loader, args)
#     for i in range(args.n_seen): 
#         prob_mean = torch.mean(all_probs[all_labels==i])
#         prob_var = torch.std(all_probs[all_labels==i])
#         threshold[i] == prob_mean - prob_var * 2
    
#     return threshold 

            