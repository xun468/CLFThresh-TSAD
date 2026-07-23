import torch
import torch.nn as nn
import numpy as np
from utils.utils import kl_recon_loss, flatten, ROC
import copy 
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

default_dim = {
'synthetic': 16,
}

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )        

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        self.meanfc = nn.Linear(hidden_dim, latent_dim)
        self.varfc = nn.Linear(hidden_dim, latent_dim)

    def reparametize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std).to(device)

        z = mu + noise * std
        return z

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape

        encoded = self.encoder(x)

        # extract latent variable z(hidden space to latent space)
        mean = self.meanfc(encoded)
        logvar = self.varfc(encoded)
        z = self.reparametize(mean, logvar)  # batch_size x latent_size        

        decoded = self.decoder(z)
        return decoded, mean, logvar, z  
    
    
def train(model, optimizer, train_loader, loss_fn):
    model.train()
    losses = [] 

    for batch in train_loader:
        x = batch.float().to(device)
        
        y_hat, m, v,l = model(x) 
        loss = loss_fn(x, y_hat, m, v)        

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

        y_hat, m, v, l = model(x) 
        loss = loss_fn(x, y_hat, m, v)
        
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

            y_hat, m, v, l = model(x)       
            loss = loss_fn(x, y_hat, m, v)
  
            mse = torch.abs(torch.sub(x,y_hat))
            score = torch.mean(mse, 2)
            
            scores += score.cpu()                
            labels += y.cpu() 
            latents += l.cpu()

    return scores, labels, latents

def calibrate(model, val_loader):
    model.eval()
    scores = []    
    latents = []

    for batch in val_loader:
        x = batch.float().to(device)
        
        y_hat, m, v, l = model(x)
        
        mse = torch.abs(torch.sub(x,y_hat))
        score = torch.mean(mse, 2)
        
        scores += score.cpu()
        latents += l.cpu()
        
    latents = torch.stack(latents)
    latents = latents.reshape(-1,3).detach().numpy()
        
    return scores, latents


def train_and_calibrate_vae(train_loader, val_loader, test_loader, args):
    model_name = "vae" 
    print("Training " + model_name)

    hidden_dim = args['hidden_dim']
    if  hidden_dim == 'default':
        hidden_dim = default_dim[args['dataset']]
    latent_dim = 3

    model = VAE(args['input_dim'],hidden_dim,latent_dim).to(device)
    loss_fn = kl_recon_loss
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