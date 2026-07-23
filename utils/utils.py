from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np 
import os 
import pandas as pd 
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc

class seq_data(Dataset):
    def __init__(self, data, seq_len, labels = None):
        self.data = data
        self.labels = labels   
        self.seq_len = seq_len        
       
        if labels is not None:
            assert len(labels) == len(data), "Labels length " + str(len(labels)) + " does not match data " + str(len(data))        
        
    def __getitem__(self, index):
        end = index + self.seq_len
        
        if self.labels is None:
            return self.data[index:end]        
         
        return self.data[index:end], self.labels[index:end]       

    def __len__(self):
        if len(self.data) < self.seq_len: 
            return 1 
        
        return len(self.data) - self.seq_len - 1  
        
def kl_recon_loss(x, recon, mean, var, kl_term = 1):
    r_loss = torch.nn.functional.mse_loss(recon, x)
    kl = - 0.5 * torch.mean(1+ var - mean.pow(2) - var.exp())
    
    return r_loss + kl_term*kl
    
def save_metrics(metrics, args):    
    if os.path.isfile(args['experiment_dir'] + "/metrics.csv"):
        metrics_df = pd.read_csv(args['experiment_dir'] + "/metrics.csv")
        if args['verbose']:
            print("reading previously saved metrics")
    else:    
        metrics_df = pd.DataFrame(columns = ["model", "entity", "metric", "score"])
        print("starting new metrics records") 

    metrics_df = pd.concat([metrics_df, pd.DataFrame(metrics)], ignore_index = True, join = "outer")
    metrics_df.drop_duplicates(inplace = True)
    metrics_df.to_csv(args['experiment_dir'] + "/metrics.csv", index = False)

    return metrics_df

def ROC(y_test, y_pred):     
    if False in np.isfinite(y_pred):
        print("infinity detected")
        y_pred = np.nan_to_num(y_pred)
        
    fpr,tpr,thresholds=roc_curve(y_test,y_pred)
    gmeans = np.sqrt(tpr * (1-fpr))
    idx = np.argmax(gmeans)
    auc_roc =roc_auc_score(y_test,y_pred)
        
    return thresholds[idx], auc

def PRC(y_test, y_pred):
    if False in np.isfinite(y_pred):
        print("infinity detected")
        y_pred = np.nan_to_num(y_pred)
        
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    auc_prc = auc(recall, precision)

    return auc_prc


def make_window(data, seq_len, labels = None):
    data_w = []
    labels_w = []
    
    if labels is None: 
        for i in range(len(data) - seq_len - 1):
            end = i + seq_len
            data_w.append(data[i:end])
        return data_w
    
    for i in range(len(data) - seq_len - 1):
        end = i + seq_len
        data_w.append(data[i:end])
        labels_w.append(labels[i:end])
        
    return data_w, labels_w

def flatten(xss):
    return [x for xs in xss for x in xs] 
    