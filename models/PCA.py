from utils.utils import make_window, flatten, ROC
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import numpy as np

def pca_test(pca, inputs):
    scores = []
    latents = [] 
    
    for series in inputs:
        series = np.array(series)
        
        series_pca = pca.transform(series)
        recon = pca.inverse_transform(series_pca)
        error = (np.absolute(series - recon)).mean(axis=1)        
        scores.append(error)   
        latents.append(series_pca)
        
    return scores, latents

def train_and_calibrate_pca(train_data, val_data, test_data, test_labels, args):
    model_name = 'pca'

    print("Evaluating " + model_name)
    calibration_data_w = make_window(val_data, args['seq_len'])
    test_data_w, labels_f = make_window(test_data, args['seq_len'], test_labels)

    pca = PCA(n_components="mle")
    _ = pca.fit_transform(train_data)      
    scores_f, latents_f = pca_test(pca, test_data_w) 
    calibration_nc,calibration_latents = pca_test(pca, calibration_data_w)
    latent_dim = len(calibration_latents[0][0])

    calibration_nc = np.array(flatten(calibration_nc))
    calibration_latents = np.array(flatten(calibration_latents))
    scores = np.array(flatten(scores_f))
    labels = np.array(flatten(labels_f))
    latents = np.array(flatten(latents_f))
    
    return scores, labels, latents, calibration_nc, calibration_latents