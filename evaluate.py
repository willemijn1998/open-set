import torch 
import numpy as np
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
numpy2ri.activate()
mvt = importr('mvtnorm')
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support

def multivariateGaussian(W, mean, cov):
    """
    Calculates the probability of input X belonging to the gaussian distribution with mean mu and covariance sigma.
    Args: 
        W: class vector W from image X, shape: n x d
        Mu:  mean of class conditional gaussian distribution, shape: 1 x d 
        Sigma: Covariance matrix of class conditional gaussian distribution, shape d x d 
    """
    d = W.shape[1]
    delta = W - mean
    distSq = (torch.matmul(delta, torch.linalg.inv(cov)) * delta).sum(axis=1)
    prob = torch.exp(-distSq/2) / ((2 * torch.pi)**(d/2) * torch.sqrt(torch.linalg.det(cov)))
    return float(prob)

def multivariateGaussianNsigma(sigma,threshold):
    """
    Qmv function gets quantile q for which P(-q <= X <= q) = threshold. 
    Args: 
        sigma: covariance matrix of a class, shape: d x d 
        threshold: (INT) percentage of train data that falls within the distribution.
    Returns: 
        p 
    """
    dim = sigma.shape[0]
    q = np.array(mvt.qmvnorm(threshold, sigma = sigma, tail = "both")[0]) 
    n = q[0]

    m = (np.diag(sigma) ** 0.5) * n
    d = (np.mat(m) * np.mat(np.linalg.pinv(sigma)) * (np.mat(m).T)) 
    p = np.exp(d * (-0.5)) / (((2 * np.pi) ** int(dim/2)) * (np.linalg.det(sigma)) ** (0.5))
    return p


def evaluate_model(model, loader, rec_threshold, classfeats, args): 
    """ 
    This function evaluates the model with given rec threshold and class features.
    Args: 
        model: model to evaluate 
        loader: data loader to evaluate on (test, validation, train)
        rec_threshold: learned reconstruction error threshold for OSR 
        classfeats: class with the means and covariance matrices per class  
        args: other arguments 
    Returns: 
        AUROC: Area under receiver operator curve
        FPR: False positive rate  
        TPR: True positive rate

    """
    true_preds, count = 0., 0 
    predictions = np.array([])
    targets = np.array([])
    for i, (input, labels) in loader: 
        labels_onehot = F.one_hot(labels)

        W_mean, Y_pred, x_rec = model.test(input, labels_onehot, args)

        # If reconstruction loss is above threshold: predict unseen 
        rec_loss = (x_rec - input).pow(2).sum((3, 2, 1))
        Y_pred[(rec_loss > rec_threshold)] = args.n_seen

        # If gaussian distance above threshold: predict unseen
        n = input.shape[0]
        delta = torch.zeros(n, args.n_seen)
        cov_matrix = classfeats.get_cov() # shape: k x len_W x len_W
        means = classfeats.get_mean() # shape: k x len_W
        for i in range(args.n_seen): 
            pNsigma = multivariateGaussianNsigma(cov_matrix[i], args.threshold_gauss)
            p = multivariateGaussian(W_mean, means[i], cov_matrix[i])
            delta[:,i] = p - pNsigma
        Y_pred[torch.logical_not(torch.any(delta>0, 1))] = args.n_seen     

        # Count number of correct predictions 
        predictions = np.append(predictions, Y_pred)
        targets = np.append(targets, labels)
        true_preds += (Y_pred.argmax(dim=1) == labels).sum().item()
        count += labels.shape[0]

    acc = true_preds / count 
    precision, recall, fscore = precision_recall_fscore_support(targets, predictions, average='macro')

    print("Precision: {} Recall: {} F-score: {}" %precision, recall, fscore)
    

    return precision, recall, fscore




    
            