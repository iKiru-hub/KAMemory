import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models import Autoencoder, MTL, logger, load_session
import utils



""" settings """

info, autoencoder = load_session(idx=0)

dim_ei = info["dim_ei"]
dim_ca3 = info["dim_ca3"]
dim_ca1 = info["dim_ca1"]
dim_eo = info["dim_eo"]

K_lat = info["K_lat"]
beta = info["beta"]
K = info["K"]

logger("<<< Loaded session >>>")


""" data """

all_num_samples = list(range(1, 20, 1))
num_datasets = len(all_num_samples)
datasets = []

for x in all_num_samples:
    training_samples_btsp = utils.sparse_stimulus_generator(N=x,
                                                            K=K,
                                                            size=dim_ei,
                                                            plot=False)

    # Convert numpy array to torch tensor
    data_tensor = torch.tensor(training_samples_btsp,
                               dtype=torch.float32)

    # Create a dataset and data loader
    dataloader = DataLoader(TensorDataset(data_tensor),
                            batch_size=1,
                            shuffle=False)

    datasets.append(dataloader)

# get weights from the autoencoder
W_ei_ca1, W_ca1_eo = autoencoder.get_weights()


"""
search 1
------

parameters to vary:
- K_lat
- K_out
- beta
"""

if bool(0):

    # settings
    num_var = 20
    var_beta = np.around(np.linspace(1, 100, num_var))
    var_K_lat = np.linspace(1, dim_ca1-2, num_var).astype(int)

    results = np.empty((num_var, num_datasets))
    results2 = np.empty((num_var, num_datasets))

    for i, dataset_i in enumerate(datasets):
        logger(f"Dataset {i+1}/{num_datasets}")
        for j, (beta_j, klat_j) in enumerate(zip(var_beta, var_K_lat)):

            # --- vary beta ---
            # make model
            model = MTL(W_ei_ca1=W_ei_ca1,
                        W_ca1_eo=W_ca1_eo,
                        dim_ca3=dim_ca3,
                        lr=1.,
                        K_lat=K_lat,
                        K_out=K,
                        beta=beta_j)

            # train
            loss_mtl, _ = utils.testing(data=dataset_i,
                                        model=model,
                                        column=True)
            results[j, i] = loss_mtl

            # --- vary K_lat ---
            # make model
            model = MTL(W_ei_ca1=W_ei_ca1,
                        W_ca1_eo=W_ca1_eo,
                        dim_ca3=dim_ca3,
                        lr=1.,
                        K_lat=klat_j,
                        K_out=K,
                        beta=beta)

            # train
            loss_mtl, _ = utils.testing(data=dataset_i,
                                        model=model,
                                        column=True)
            results2[j, i] = loss_mtl



    logger("<<< Search done >>>")


    """ plot """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("MSE loss")

    cb1 = ax1.imshow(results, cmap="viridis", aspect="auto",
                     vmin=0.)
    # bar
    cbar1 = fig.colorbar(cb1, ax=ax1)

    ax1.set_xlabel("Number of samples")
    ax1.set_xticks(range(num_datasets))
    ax1.set_xticklabels(all_num_samples)
    ax1.set_ylabel("$\\beta$")
    ax1.set_yticks(range(num_var))
    ax1.set_yticklabels(var_beta)
    ax1.set_title("Varying $\\beta$, $K_{lat}=$" + f"{K_lat}")

    cb2 = ax2.imshow(results2, cmap="viridis", aspect="auto",
                     vmin=0.)
    # bar
    cbar2 = fig.colorbar(cb2, ax=ax2)

    ax2.set_xlabel("Number of samples")
    ax2.set_xticks(range(num_datasets))
    ax2.set_xticklabels(all_num_samples)
    ax2.set_ylabel("$K_{lat}$")
    ax2.set_yticks(range(num_var))
    ax2.set_yticklabels(var_K_lat)
    ax2.set_title("Varying $K_{lat}$" + f", $\\beta={beta}$")

    plt.tight_layout()

    plt.show()

else:

    # settings
    num_var = 20
    idx_dataset = 2
    var_beta = np.around(np.linspace(1, 100, num_var))
    var_K_lat = np.linspace(1, dim_ca1-2, num_var).astype(int)

    results = np.empty((num_var, num_var))

    for (i, beta_i) in tqdm(enumerate(var_beta)):
        for j, klat_j in enumerate(var_K_lat):

            # --- vary beta ---
            # make model
            model = MTL(W_ei_ca1=W_ei_ca1,
                        W_ca1_eo=W_ca1_eo,
                        dim_ca3=dim_ca3,
                        lr=1.,
                        K_lat=klat_j,
                        K_out=K,
                        beta=beta_i)

            # train
            loss_mtl, _ = utils.testing(data=datasets[idx_dataset],
                                        model=model,
                                        column=True)
            results[j, i] = loss_mtl


    logger("<<< Search done >>>")

    """ plot """

    plt.imshow(results, cmap="viridis", aspect="auto",
               vmin=0.)
    plt.colorbar()
    plt.xlabel("$\\beta$")
    plt.xticks(range(num_var), var_beta)
    plt.ylabel("$K_{lat}$")
    plt.yticks(range(num_var), var_K_lat)

    plt.title(f"MSE loss | samples={all_num_samples[idx_dataset]}")
    plt.show()



