import numpy as np 
import pandas as pd 
from sklearn import preprocessing
import os 

valid_entities = {
"synthetic" : ["mixed", "shapelet", "seasonal", "trend", "point_context", "point_global"],
}

def read_synthetic(entity = "mixed"):
    if entity == None: 
        entity = "mixed" 
    
    print("Getting entity " + entity) 
    #sanity check 
    assert entity in valid_entities["synthetic"], entity + " is not part of the synthetic dataset" 
    
    normal = pd.read_csv("datasets/synthetic/train.csv")
    attack = pd.read_csv("datasets/synthetic/test_" + entity + ".csv") 
    labels = attack['anomaly']
    attack = attack.drop("anomaly", axis = 1)
    
    return normal, attack, labels  

get_datasets = {
"synthetic" : read_synthetic, 
}

def get_data(dataset, val_split=0.2, seq_len=12, down_rate=1, entity = None, verbose = False):    
    if entity: 
        normal, attack, labels = get_datasets[dataset](entity)
    else:
        normal, attack, labels = get_datasets[dataset]()           
  
    if down_rate > 1:
        #Downsampling
        normal=normal.groupby(np.arange(len(normal.index)) // down_rate).mean()
        attack=attack.groupby(np.arange(len(attack.index)) // down_rate).mean()
        labels = pd.DataFrame(labels)    
        labels = labels.groupby(np.arange(len(labels.index)) // down_rate).max()   
        labels = labels.values.flatten()
    else:        
        labels = np.array(labels)
        

    #Normalizing     
    min_max_scaler = preprocessing.MinMaxScaler()
    normal = min_max_scaler.fit_transform(normal.values)
    attack = min_max_scaler.transform(attack.values)       

    split = int(len(normal) * (1-val_split))
    validate = normal[split:]
    normal = normal[:split]
    
    if verbose:
        print(normal.shape)
        print(validate.shape)
        print(attack.shape)
        print(labels.shape)

    return normal, validate, attack, labels
