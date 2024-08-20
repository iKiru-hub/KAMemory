import numpy as np
import torch
from models import Autoencoder, logger
import utils




""" settings """

# architecture sizes
dim_ei = 50
dim_ca3 = 50 
dim_ca1 = 50
dim_eo = dim_ei

# data settings
num_samples = 300
num_reconstructions = 1
K = 5

training_samples = utils.sparse_stimulus_generator(N=num_samples,
                                                   K=K,
                                                   size=dim_ei,
                                                   plot=False)
test_samples = utils.sparse_stimulus_generator(N=num_samples,
                                               K=K,
                                               size=dim_ei,
                                               plot=False)

logger(f"Training data generated: {training_samples.shape}")


""" autoencoder training """

K_lat = 15
beta = 60
autoencoder = Autoencoder(input_dim=dim_ei,
                          encoding_dim=dim_ca1,
                          activation=None,
                          K=K_lat,
                          beta=beta)
logger(f"%Autoencoder: {autoencoder}")

# train autoencoder
epochs = 100
loss_ae, autoencoder = utils.train_autoencoder(
                training_data=training_samples,
                test_data=test_samples,
                model=autoencoder,
                epochs=int(epochs),
                batch_size=5, learning_rate=1e-3)

logger(f"<<< Autoencoder trained [loss={loss_ae:.4f}] >>>")


""" save """

info = {
    "dim_ei": dim_ei,
    "dim_ca3": dim_ca3,
    "dim_ca1": dim_ca1,
    "dim_eo": dim_eo,
    "num_samples": num_samples,
    "num_reconstructions": num_reconstructions,
    "K": K,
    "K_lat": K_lat,
    "beta": beta,
    "epochs": epochs,
    "loss_ae": round(loss_ae, 5),
}

# make directory
import os
import datetime
import json

dir_name = "cache/ae_" + datetime.datetime.now().strftime("%d_%H%M")

os.makedirs(dir_name, exist_ok=True)

# save model
torch.save(autoencoder.state_dict(), f"{dir_name}/autoencoder.pt")

# save info
with open(f"{dir_name}/info.json", "w") as f:
    json.dump(info, f)

logger(f"<<< Model saved in {dir_name} >>>")
