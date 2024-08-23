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

alpha = 0.01

# get weights from the autoencoder
W_ei_ca1, W_ca1_eo = autoencoder.get_weights()

logger("<<< Loaded session >>>")


""" data """

num_samples = 100
num_rep = 5
datasets = []

stimuli = utils.sparse_stimulus_generator(N=num_samples,
                                          K=K,
                                          size=dim_ei,
                                          plot=False)

datasets = []
for k in range(num_samples):
    data = torch.tensor(stimuli[:k+1], dtype=torch.float32)
    dataloader = DataLoader(TensorDataset(data),
                            batch_size=1,
                            shuffle=False)
    datasets += [dataloader]

"""

[[s1]
 [s1, s2]
 ...]

"""


""" run """

num_alphas = 1
if num_alphas < 2:
    alphas = [0.4]
else:
    alphas = np.around(np.linspace(0.075, 0.3, num_alphas), 2)

outputs = np.zeros((num_rep, num_alphas, num_samples, num_samples))

for l in tqdm(range(num_rep)):

    for h, alpha in enumerate(alphas):

        # data
        stimuli = utils.sparse_stimulus_generator(N=num_samples,
                                                  K=K,
                                                  size=dim_ei,
                                                  plot=False)

        datasets = []
        for k in range(num_samples):
            data = torch.tensor(stimuli[:k+1], dtype=torch.float32)
            dataloader = DataLoader(TensorDataset(data),
                                    batch_size=1,
                                    shuffle=False)
            datasets += [dataloader]

        # run
        for i in tqdm(range(num_samples)):

            # make model
            model = MTL(W_ei_ca1=W_ei_ca1,
                        W_ca1_eo=W_ca1_eo,
                        dim_ca3=dim_ca3,
                        lr=1.,
                        K_lat=K_lat,
                        K_out=K,
                        beta=beta,
                        alpha=alpha)

            # train
            model.eval()
            with torch.no_grad():
                for batch in datasets[i]:
                    # forward
                    _ = model(batch[0].reshape(-1, 1))

            # test
            model.pause_lr()
            model.eval()
            with torch.no_grad():
                for j, batch in enumerate(datasets[i]):
                    x = batch[0].reshape(-1, 1)

                    # forward
                    y = model(x)
                    # logger.debug(f"{x.shape}, {y.shape}")

                    # record : cosine similarity
                    # outputs[l, h, i, j] = (y.T @ x) / \
                    #     (torch.norm(x) * torch.norm(y))

                    value = (y.T @ x) / \
                        (torch.norm(x) * torch.norm(y))

                    outputs[l, h, i, j] = (value.item() - 0.2) / 0.8


""" plot """

plot_type = 3

if plot_type == 0:

    plt.figure()

    plt.subplot(221)
    orout = outputs.copy()
    outputs = outputs.mean(axis=0).sum(axis=0)
    plt.imshow(outputs, cmap="viridis",
               vmin=0, vmax=1, aspect="equal",
               interpolation="nearest")

    plt.colorbar()
    plt.xlabel("stimuli in a run")
    # plt.xticks(range(num_samples), range(1, num_samples+1))
    plt.ylabel("different runs")
    # plt.yticks(range(num_samples), range(1, num_samples+1))

    plt.title(f"MSE loss | $K=${K} - $K_l=${K_lat} $\\beta=${beta} $\\alpha=${alpha}")


    plt.subplot(222)

    n = 20
    nd = 100

    colors = plt.cm.rainbow(np.linspace(0, 1, nd))
    for i in range(nd):
        vidiag = np.diag(outputs[i:]).flatten()
        vidiag = np.convolve(vidiag, np.ones(n)/n, mode="same")[n//2:-n//2]
        plt.plot(vidiag, alpha=0.9/(i+1)**0.7,
                 color=colors[i])

    plt.title("distribution of diagonal elements")
    # plt.xticks(range(num_samples), range(1, num_samples+1))
    # plt.legend()
    plt.ylim(0, 1.05)
    plt.grid()

    plt.subplot(212)
    outputs = outputs.sum(axis=1)

    # plt.figure()

    plt.plot(outputs, '-k')
    plt.title(f"sum of diagonal elements | $K=${K} - $K_l=${K_lat} $\\beta=${beta} $\\alpha=${alphas}")

    plt.show()

elif plot_type == 1:

    outputs = outputs.mean(axis=0)

    plt.figure()

    # vdiag = np.diag(outputs, axis=1)
    vdiag = np.array([np.diag(outputs[l]) for l in range(num_alphas)])
    plt.imshow(vdiag, cmap="viridis", vmin=0, vmax=1, aspect="auto",
               interpolation="nearest")

    plt.yticks(range(num_alphas), alphas)
    plt.ylabel("$\\alpha$")
    plt.xlabel("stimuli in a run")
    # plt.xticks(range(num_samples), range(1, num_samples+1))

    plt.title(f"diagonal elements | $K=${K} - $K_l=${K_lat} $\\beta=${beta}")

    plt.show()

elif plot_type == 2:

    outputs = outputs.mean(axis=0).sum(axis=0).sum(axis=1)

    plt.figure(figsize=(10, 4))

    plt.plot(outputs, '-k')
    plt.title(f"sum of diagonal elements | $K=${K} - $K_l=${K_lat} $\\beta=${beta} $\\alpha=${alphas}")

    plt.show()

elif plot_type == 3:

    outputs = outputs.mean(axis=0).sum(axis=0)

    plt.figure()

    plt.subplot(121)
    plt.imshow(outputs, cmap="viridis",
               vmin=0, vmax=1, aspect="auto")

    plt.subplot(122)
    # plt.axhline(0.1, color="r", linestyle="--",
    #             alpha=0.2)
    # smoothing
    outputs = outputs[:, 0]
    nsmooth = 10
    outputs = np.convolve(outputs,
                          np.ones(nsmooth)/nsmooth,
                          mode="same")
    plt.plot(outputs, 'o-')

    plt.ylim(0., 1)
    plt.grid()

    plt.show()



