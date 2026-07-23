import torch
import torch.nn as nn
import numpy as np
from utils.utils import flatten, ROC
import copy 
from sklearn.neighbors import NearestNeighbors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
from tqdm import tqdm

default_dim = {
'synthetic' : 16,
}

class AE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )        

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded   
    
def train(model, optimizer, train_loader, loss_fn):
    model.train()
    losses = [] 

    for batch in train_loader:
        x = batch.float().to(device)
        
        y_hat, _ = model(x)
        loss = loss_fn(x, y_hat)        

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses += [loss.item()]
    
    return losses 

def val(model, val_loader, loss_fn):
    model.eval()
    losses = []    

    for batch in val_loader:
        x = batch.float().to(device)
        
        y_hat, _ = model(x)
        loss = loss_fn(x, y_hat) 
        
        losses += [loss.item()]
        
    return np.mean(losses)

def test(model, test_loader, loss_fn):      
    with torch.no_grad():
        labels = []  
        scores = []
        latents = []
        
        model.eval()
        for batch in test_loader:
            x, y = batch[0].float().to(device), batch[1].float().to(device)
        
            y_hat, l = model(x)
            loss = loss_fn(x, y_hat)
        
            # mse = nn.functional.mse_loss(y, y_hat, reduction='none').cpu()
            mse = torch.abs(torch.sub(x,y_hat))
            score = torch.mean(mse, 2)

            scores += score.cpu()               
            labels += y.cpu()
            latents += l.cpu()
    
    # print(len(scores))
    # print(len(labels))

    return scores, labels, latents 

def calibrate(model, val_loader):
    model.eval()
    scores = []    
    latents = []

    for batch in val_loader:
        x = batch.float().to(device)
        
        y_hat, l = model(x)
        
        mse = torch.abs(torch.sub(x,y_hat))
        score = torch.mean(mse, 2)
        
        scores += score.cpu()
        latents += l.cpu()
        
    latents = torch.stack(latents)
    latents = latents.reshape(-1,3).detach().numpy()
        
    return scores, latents

def train_and_calibrate_ae(train_loader, val_loader, test_loader, args):
    model_name = "ae" 
    print("Training " + model_name)

    hidden_dim = args['hidden_dim']
    if  hidden_dim == 'default':
        hidden_dim = default_dim[args['dataset']]
    latent_dim = 3 

    model = AE(args['input_dim'],hidden_dim,latent_dim).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    best_val = 10000
    for i in tqdm(range(args['num_epochs'])):
        train(model, optimizer, train_loader, loss_fn)
        val_losses = val(model, val_loader, loss_fn)
        if val_losses < best_val:
            best_val = val_losses 
            best_model_state_dict = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), args['experiment_dir'] + "/" + model_name + ".pth")

    model.load_state_dict(torch.load(args['experiment_dir'] + "/" + model_name + ".pth"))
    scores_f, labels_f, latents_f = test(model, test_loader, loss_fn)
    calibration_nc, calibration_latents = calibrate(model, val_loader)
    calibration_nc = torch.stack(calibration_nc).reshape(-1).detach().numpy()

    scores = torch.stack(scores_f).reshape(-1).detach().numpy()
    labels = torch.stack(labels_f).reshape(-1).detach().numpy()
    latents = torch.stack(latents_f).reshape(-1,latent_dim).detach().numpy()
    
    return scores, labels, latents, calibration_nc, calibration_latents