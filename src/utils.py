import numpy as np
import matplotlib.pyplot as plt
import warnings

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

from numba import jit


from tqdm import tqdm
import os
import json

import models
from logger import logger

# print(f"{__name__}::{ISCOLORED=}")

MEDIA_PATH = "".join((os.getcwd().split("KAMemory")[0], "KAMemory/media/"))



""" MISCELLANOUS """


def tqdm_enumerate(iter, **tqdm_kwargs):
    i = 0
    for y in tqdm(iter, **tqdm_kwargs):
        yield i, y
        i += 1


def calc_capacity(outputs: np.ndarray,
                  threshold: float,
                  nsmooth: int=20,
                  idx_pattern: int=None) -> int:

    """
    This function calculates the capacity of the network
    by finding the number of patterns that can be stored.

    Parameters
    ----------
    outputs : np.ndarray
        outputs of the network
    threshold : float
        threshold value
    nsmooth : int, optional
        smoothing factor, by default 20
    idx_pattern : int, optional
        index of the pattern, by default None

    Returns
    -------
    int or list
        capacity
    """

    assert outputs.ndim == 2, "outputs must be 2D"

    if idx_pattern is None:

        results = []
        for i in range(outputs.shape[0]):
            idx = calc_capacity(outputs=outputs,
                                threshold=threshold,
                                nsmooth=nsmooth,
                                idx_pattern=i)
            results.append(idx)

        return results

    # select the first pattern and pad it
    padded_out = np.pad(outputs[idx_pattern:, idx_pattern],
                        (nsmooth-1, 0), mode="edge")

    # smooth it
    outputs = np.convolve(padded_out,
                      np.ones(nsmooth)/nsmooth,
                      mode="valid")

    # find the highest index where the output
    # is below the threshold
    idx = np.argmin(np.where(outputs >= threshold,
                             outputs, -np.inf),
                    axis=0).item()

    return idx



""" visualization """


def plot_squashed_data(data: np.ndarray, title: str="",
                       ax: plt.Axes=None, squash: bool=False,
                       proper_title: bool=False):

    """
    This function plots the squashed data

    Parameters
    ----------
    data : np.ndarray
        squashed data
    title : str, optional
        title of the plot, by default ""
    ax : plt.Axes, optional
        axis of the plot, by default None
    """

    if squash:
        data = data.sum(axis=0).reshape(1, -1)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.imshow(data, aspect="auto", cmap="gray_r", vmin=0, vmax=1)
    if proper_title:
        ax.set_title(title, fontsize=15)
    else:
        ax.set_ylabel(title, fontsize=15)
    ax.set_yticks(range(len(data)), range(1, 1+len(data)))
    ax.set_xticks([])

    if ax is None:
        plt.show()


def plot_input(input, network_params):
  mec = input[:network_params["dim_mec"]]
  lec = input[network_params["dim_mec"]:]

  fig, axs = plt.subplots(1, 2, figsize=(12, 12))
  axs[0].imshow(mec.reshape(network_params["mec_N_y"], network_params["mec_N_x"]), cmap='gray_r')
  axs[0].set_title('space')
  lec_width = int(np.sqrt(network_params["dim_lec"]))
  axs[1].imshow(lec.reshape((-1, network_params["dim_lec"])), cmap='gray_r')
  axs[1].set_title('sensory')
  axs[0].set_yticks([])
  axs[1].set_yticks([])
  axs[0].set_xticks([])
  axs[1].set_xticks([])
  plt.show()



""" activation funcitons """



class SparsemaxFunction(autograd.Function):

    @staticmethod
    def forward(ctx, z: torch.Tensor) -> torch.Tensor:

        """
        Parameters
        ----------
        z : torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            output tensor
        """

        z = z.clone()
        dim = -1  # Assuming last dimension for sparsemax
        sorted_z, _ = torch.sort(z, descending=True, dim=dim)
        cumsum_sorted = sorted_z.cumsum(dim)
        k = torch.arange(1, z.size(dim) + 1).to(z.device)
        support = k * sorted_z > (cumsum_sorted - 1)
        k_z = torch.sum(support, dim=dim, keepdim=True).float()
        tau_z = (cumsum_sorted - 1) / k_z
        output = torch.clamp(z - tau_z, min=0)
        ctx.save_for_backward(output)

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:

        """
        Parameters
        ----------
        grad_output : torch.Tensor
            gradient tensor

        Returns
        -------
        torch.Tensor
            gradient tensor
        """

        output, *_ = ctx.saved_tensors

        nonzeros = torch.ne(output, 0)
        support_size = nonzeros.sum(dim=-1, keepdim=True)
        v_hat = (grad_output * nonzeros).sum(-1,
                            keepdim=True) / support_size

        return nonzeros * (grad_output - v_hat)


class Sparsemax(nn.Module):

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return SparsemaxFunction.apply(z)


class Identity(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class SoftSigmoidFunction(autograd.Function):

    @staticmethod
    def forward(ctx, z: torch.Tensor,
                gamma: float=1., beta: float=1.,
                alpha: float=0.) -> torch.Tensor:

        """
        Parameters
        ----------
        z : torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            output tensor
        """

        ctx.save_for_backward(z)

        exp_z = torch.exp(gamma * z)
        z_softmax = exp_z / exp_z.sum(dim=-1, keepdim=True)

        z_sigmoid = 1 / (1 + torch.exp(
            -beta * (z_softmax - alpha)))

        ctx.save_for_backward(z_softmax)
        ctx.save_for_backward(z_sigmoid)

        ctx.gamma = gamma
        ctx.beta = beta
        ctx.alpha = alpha

        return z_sigmoid

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:

        """
        Parameters
        ----------
        grad_output : torch.Tensor
            gradient tensor

        Returns
        -------
        torch.Tensor
            gradient tensor
        """

        z, z_softmax, z_sigmoid = ctx.saved_tensors
        print("length: ", len(ctx.saved_tensors))

        # --- calc jacobian [for the softmax] ---
        s_diag = torch.diag_embed(z_softmax)

        # Calculate the outer product of the softmax vector with itself
        s_outer = torch.einsum('bi,bj->bij', s, s)

        # Compute the Jacobian matrix
        jacobian = s_diag - s_outer

        # --- calc grad ---
        grad = ctx.gamma * ctx.beta * \
            (z_sigmoid * (1 - z_sigmoid)) * \
            torch.einsum('bi,bij->bj', grad_output, jacobian)

        return grad


class SoftSigmoid(nn.Module):

    def __init__(self, gamma: float=1.,
                 beta: float=1., alpha: float=0.):

        super(SoftSigmoid, self).__init__()

        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return SoftSigmoidFunction.apply(z, self.gamma,
                                         self.beta, self.alpha)


def sparsemoid(z: torch.Tensor, K: int,
               beta: float, flag=False) -> torch.Tensor:

    if K > 0:
        z_sorted = torch.sort(z, descending=True, dim=1).values

        alpha = z_sorted[:, K-1: K+1]
        alpha = alpha.mean(axis=1).reshape(-1, 1)

    # apply
    z = beta * (z - alpha)
    return torch.sigmoid(z)


def cross_entropy(x: torch.Tensor, y: torch.Tensor, eps=1e-8):
    return F.binary_cross_entropy(x, y)

def cosine_similarity_vec(x, y):
    return (y.T @ x) / (torch.norm(x) * torch.norm(y))

def cosine_similarity_mat(matrix1: np.ndarray, matrix2: np.ndarray):
    """
    Compute the normalized dot product (cosine similarity) between two matrices.

    Parameters:
    matrix1 (numpy.ndarray): First matrix with shape (m, n)
    matrix2 (numpy.ndarray): Second matrix with shape (m, p)

    Returns:
    numpy.ndarray: Cosine similarity matrix with shape (n, p)
    """
    # Compute the dot product
    dot_product = matrix1.T @ matrix2  # Shape: (n, p)

    # Compute the norms (Frobenius norm) for each column
    norm1 = np.linalg.norm(matrix1, axis=0).reshape(-1, 1)  # Shape: (n, 1)
    norm2 = np.linalg.norm(matrix2, axis=0).reshape(1, -1)  # Shape: (1, p)

    # Compute the outer product of norms
    norm_product = norm1 @ norm2  # Shape: (n, p)

    # Normalize the dot product by dividing by the norm product
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-8
    cosine_sim = dot_product / (norm_product + epsilon)

    return cosine_sim


""" stimulus generator """

def stimulus_generator(N: int, size: int=10, heads: int=2, variance: float=0.1,
                       higher_heads: int=None, higher_variance: float=None,
                       plot: bool=False, use_uniform: bool=True) -> np.ndarray:

    """
    This function generates random z patterns with a certain
    degree of structure

    Parameters
    ----------
    N : int
        Number of samples
    size : int, optional
        Size of the z patterns, by default 10
    # generate docstring
        heads : int, optional
        Number of heads, by default 2
    variance : float, optional
        Variance of the Gaussian used to generate the z patterns, by default 0.1
    higher_heads : int, optional
        Higher number of heads, by default None
    higher_variance : float, optional
        Higher variance of the Gaussian used to generate the z patterns, by default None
    plot : bool, optional
        Whether to plot the z patterns, by default False

    Returns
    -------
    samples : np.ndarray
        z patterns
    """

    # generate the position of the heads drawing from a distribution defined
    # by higher_heads and higher_variance
    if higher_heads is not None and higher_variance is not None:
        if higher_heads != heads:
            warnings.warn("higher_heads must be equal to heads, setting higher_heads = heads")
            higher_heads = heads

        high_mu = np.linspace(1/(higher_heads+1), 1 - 1/(higher_heads+1),
                              higher_heads, endpoint=True) * size
        high_variance = np.array([higher_variance]*heads) * size

        # generate the positions of the heads
        mu = np.zeros((N, heads))
        for i in range(N):
            if use_uniform:
                mu[i, :] = np.random.choice(range(size), replace=False, size=heads)
            else:
                for k, (hh, hv) in enumerate(zip(high_mu, high_variance)):
                    np.random.normal(hh, hv)
        variance = np.array([variance]*heads) * size

        # tile for the number of samples
        variance = np.tile(variance, (N, 1))

    # generate the position of the heads as equidistant points
    else:
        mu = np.linspace(1/(heads+1), 1 - 1/(heads+1), heads, endpoint=True) * size
        variance = np.array([variance]*heads) * size

        # tile for the number of samples
        mu = np.tile(mu, (N, 1))
        variance = np.tile(variance, (N, 1))

    # generate the z patterns
    samples = np.zeros((N, size))
    for i in range(N):
        for k in range(heads):
            for x in range(size):
                p = np.exp(-((x-mu[i, k])**2)/(2*variance[i, k]))
                samples[i, x] += np.random.binomial(1, p)

    if plot:
        plot_stimuli(samples=samples)

    return samples


def sparse_stimulus_generator(N: int, K: int,
                              size: int=10, plot: bool=False) -> np.ndarray:

    """
    This function generates random z patterns with a certain
    degree of sparsity

    Parameters
    ----------
    N : int
        Number of samples
    K : int
        Number of active units
    size : int, optional
        Size of the z patterns, by default 10
    plot : bool, optional
        Whether to plot the z patterns.
        Default False

    Returns
    -------
    samples : np.ndarray
        z patterns
    """

    samples = np.zeros((N, size))
    for i in range(N):
        idx = np.random.choice(range(size), replace=False, size=K)
        samples[i, idx] = 1

    samples = samples.astype(np.float32)

    if plot:
        plot_stimuli(samples=samples)

    return samples


def plot_stimuli(samples: np.ndarray):

    """
    This function plots the z patterns

    Parameters
    ----------
    samples : np.ndarray
        z patterns
    """

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    ax.imshow(samples, aspect="auto", vmin=0, vmax=1, cmap="gray_r")
    ax.set_xlabel("size")
    ax.set_ylabel("samples")
    if len(samples) < 20:
        ax.set_yticks(range(samples.shape[0]))
        ax.set_yticklabels(range(1, 1+samples.shape[0]))
    ax.set_title("z patterns")

    ax2.imshow(samples.sum(axis=0).reshape(1, -1),
               aspect="auto", cmap="gray_r")
    ax2.set_xlabel("size")
    ax2.set_title("Average z pattern")
    plt.show()


def sparse_stimulus_generator_sensory(num_stimuli: int, K : int,
                                      mec_size: int,  lec_size: int,
                                      N_x : int, N_y : int,
                                      pf_sigma: int,
                                      num_laps: int,
                                      lap_length: int=None,
                                      num_cues: int=None,
                                      position_list=None,
                                      cue_positions=None,
                                      sen_list=None,
                                      plot: bool=False,
                                      sigma: float=5,
                                      verbose: bool=True,
                                      binarize: bool=False) -> np.ndarray:

    """
    This function generates random z patterns with a certain
    degree of sparsity

    Parameters
    ----------
    N : int
        Number of samples
    K : int
        Number of active units
    size : int, optional
        Size of the z patterns, by default 10
    plot : bool, optional
        Whether to plot the z patterns.
        Default False

    Returns
    -------
    samples : np.ndarray
        z patterns
    """

    def place_field_activity(N_x, N_y, sigma, xi, yi):
        """
        Computes place field activity for each cell on an NxN grid for a given location (xi, yi).
        """

        def circular_distance(x1, x2, N):
            """
            Computes the minimum circular distance in the x-direction (wraps around the boundaries).
            """
            return np.minimum(np.abs(x1 - x2), N - np.abs(x1 - x2))

        # Create a grid of size NxN with place cells at each position
        x = np.linspace(0, N_x-1, N_x)
        y = np.linspace(0, N_y-1, N_y)
        X, Y = np.meshgrid(x, y)
        # Calculate the squared Euclidean distance between (xi, yi) and each place cell location
        dist_squared = circular_distance(X, xi, N_x) ** 2 + (Y - yi) ** 2

        # Compute Gaussian activity for each place cell
        activity = np.exp(-dist_squared / (2 * sigma ** 2))
        return activity

    # --- init
    IS_CUE = position_list is not None

    # --- make cue patterns
    #cue_pattern_size = lec_size//num_cues  #
    if lec_size > 0 and num_cues is not None:
        fixed_cue = np.zeros((num_cues, lec_size))
        for k in range(num_cues):
            cue_idx = np.random.choice(range(lec_size), replace=False, size=K)
            #fixed_cue[cue_idx] = 1
            fixed_cue[k, K*k: K*(k+1)] = 1  # | 1, 1, .0s  , 1, 1, ..0s.. |

    # 0, 0, 1, 0, 1, 1...
    #lap_cues = np.random.choice(range(num_cues), size=num_laps) if num_laps is not None else None
    lap_cues = np.zeros(num_laps) if position_list is not None else None
    position_pool = np.arange(N_x)  # Possible positions in 1D track
    samples = np.zeros((num_stimuli, mec_size + lec_size))
    alpha_samples = np.zeros(num_stimuli)

    lap_i = 0
    cue_duration = 1

    # --- loop over each time step in each lap
    for i in range(num_stimuli): # laps x length

        # determine lap-cue association
        if IS_CUE:

            # count laps
            if i % lap_length == 0:
                lap_idx = (lap_i // cue_duration) % num_cues
                lap_cues[lap_i] = lap_idx

                if verbose:
                    print(f"-------------\nlap {lap_i}, cue {lap_idx}")
                lap_i += 1

        # SPATIAL input
        if mec_size > 0:
            # Reset the pool every 50 samples
            if position_list is None:
              if i % mec_size == 0:
                  np.random.shuffle(position_pool)  # Shuffle the pool to get random order
              pos_idx = i % 50  # Get position index within the shuffled pool
              x_i = position_pool[pos_idx]  # Choose the x position
              y_i = np.random.randint(0, N_y)
            else:
              x_i, y_i = position_list[i]
            activity_grid = place_field_activity(N_x, N_y, pf_sigma, x_i, y_i)
            if binarize:
                activity_grid = (activity_grid > 0.5).astype(float)
            samples[i, :mec_size] = activity_grid.flatten()

        # SENSORY input
        if lec_size > 0:

            if IS_CUE:
                #p = samples[i, cue_positions[lap_cues[lap_idx]]] / \
                p = samples[i, cue_positions[lap_idx]] / \
                    samples[i, :mec_size].max()
                alpha_samples[i] = p

                if np.random.binomial(1, p):
                    activity_lec = fixed_cue[lap_cues[lap_idx].astype(int)]
                else:
                    #activity_lec = np.zeros((lec_size))
                    #lec_idx = np.random.choice(range(lec_size),
                    #                           replace=False, size=K)
                    #activity_lec[lec_idx] = 1

                    activity_lec = np.zeros((lec_size))
                    lec_idx = np.random.choice(range(lec_size),
                                               replace=False, size=K)
                    activity_lec[lec_idx] = 1
            else:
                if IS_CUE:
                    lec_idx = np.random.choice(range(lec_size),
                                               replace=False, size=K)
                    activity_lec[lec_idx] = 1

                else:
                    activity_lec = np.zeros((lec_size))
                    lec_idx = np.random.choice(range(lec_size),
                                           replace=False, size=K)
                    activity_lec[lec_idx] = 1


            samples[i, mec_size:] = sen_list[i] if sen_list is not None else activity_lec

    samples = samples.astype(np.float32) # [ 00000 |      ]

    return samples, lap_cues, alpha_samples


def get_track_input(tp: dict, network_params: dict):

  position_list = [(x, 0) for lap in range(tp["num_laps"]) for x in range(tp["length"])]

  if tp["reward"] == "random":
    reward_list = None
  if tp["cue"] == "random":
    cue_list = None

  sen_list = None

  track_input, lap_cues, alpha_samples = sparse_stimulus_generator_sensory(num_stimuli=tp["num_laps"]*tp["length"],
                                                  K = network_params["K_lec"],
                                                  mec_size=network_params["dim_mec"],
                                                  lec_size=network_params["dim_lec"],
                                                  N_x=network_params["mec_N_x"],
                                                  N_y=network_params["mec_N_y"],
                                                  pf_sigma=network_params["mec_sigma"],
                                                  lap_length=track_params["length"],
                                                  num_laps=track_params["num_laps"],
                                                  num_cues=network_params["num_cues"],
                                                  position_list=position_list,
                                                  cue_positions=tp["cue_position"],
                                                  sen_list=None,
                                                  plot=False)
  return track_input, lap_cues, alpha_samples


""" training """


def train_autoencoder(training_data: np.ndarray,
                      test_data: np.ndarray,
                      model: object,
                      epochs: int=20, batch_size: int=64,
                      learning_rate: float=1e-3):

    """
    Train the autoencoder model

    Parameters
    ----------
    training_data: np.ndarray
        z training data
    test_data: np.ndarray
        z test data
    model: nn.Module
        the autoencoder model
    epochs: int
        the number of epochs
    batch_size: int
        the batch size
    learning_rate: float
        the learning rate
    """

    # Convert numpy array to torch tensor
    data_tensor = torch.tensor(training_data, dtype=torch.float32)

    # Create a dataset and data loader
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # test data
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
    test_dataset = TensorDataset(test_data_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Loss function and optimizer
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Set the model to training mode
    model.train()

    # Training loop
    epoch = 0
    epoch_log = 100
    for epoch in (pbar := tqdm(range(epochs), desc = f"{epoch}")):
        total_loss = 0
        for batch in dataloader:
            zs = batch[0]

            # Forward pass
            outputs = model(zs)
            loss = criterion(outputs, zs)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # test loss
        test_loss, _ = testing(data=test_dataloader,
                               model=model,
                               criterion=criterion)

        if (epoch+1) % epoch_log == 0:
            pbar.set_description(f"Epoch [{epoch+1}], " + \
                f"Loss: {total_loss / len(dataloader):.4f}, " + \
                                 f"Test: {test_loss:.4f}")

    return total_loss, model


def reconstruct_data(data: np.ndarray, model: object, num: int=5,
                     column: bool=False, show: bool=True, plot: bool=True):

    """
    Reconstruct data using the autoencoder model

    Parameters
    ----------
    data: np.ndarray
        z data
    num: int
        the number of samples to reconstruct
    model: nn.Module
        the autoencoder model

    Returns
    -------
    np.ndarray
        reconstructed data
    """

    # Convert numpy array to torch tensor
    if not isinstance(data, torch.Tensor):
        data_tensor = torch.tensor(data[:num],
                                   dtype=torch.float32)
    else:
        data_tensor = data[:num].clone().detach()

    # Create a dataset and data loader
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Set the model to evaluation mode
    model.eval()
    criterion = MSELoss()

    # Reconstruct data
    reconstructed_data = []
    latent_data = []
    loss = 0.
    with torch.no_grad():

        for batch in tqdm(dataloader):

            zs = batch[0] if not column else batch[0].reshape(-1, 1)

            # Forward pass
            outputs, latent = model(zs, ca1=True)
            reconstructed_data.append(outputs.numpy().flatten())
            latent_data.append(latent.numpy().flatten())

            # evaluate the output
            loss += criterion(outputs, zs)

    # Convert list to numpy array
    reconstructed_data = np.array(reconstructed_data)

    # difference between original and reconstructed data
    diff_data = (data[:num] - reconstructed_data)

    loss /= len(dataloader)

    # plot
    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        ax1.imshow(data_tensor, aspect="auto", vmin=0, vmax=1, cmap="gray_r")
        ax1.set_title("Original data")
        ax1.set_axis_off()

        ax2.imshow(reconstructed_data, aspect="auto", vmin=0, vmax=1, cmap="gray_r")
        ax2.set_title("Reconstructed data")
        ax2.set_axis_off()

        ax3.imshow(diff_data, aspect="auto", cmap="seismic", vmin=-1, vmax=1)
        ax3.set_title(f"Difference [loss={loss:.4f}]")
        ax3.set_axis_off()

        if show:
            plt.show()

    return reconstructed_data, latent_data


def testing(data: np.ndarray, model: object,
            criterion: object=MSELoss(),
            column: bool=False,
            use_tensor: bool=False,
            progressive_test: bool=False):

    """
    Test the model

    Parameters
    ----------
    data: np.ndarray
        z data
    model: nn.Module
        the model
    """

    if not isinstance(data, DataLoader):
        # Convert numpy array to torch tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # Create a dataset and data loader
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    else:
        dataloader = data

    if use_tensor:
        try:
            data_tensor
        except NameError:
            raise ValueError("data_tensor is not defined")
        dataloader = data_tensor.unsqueeze(1)

    # Set the model to evaluation mode
    model.eval()
    loss = 0.
    acc_matrix = torch.zeros(len(dataloader), len(dataloader))

    record = []

    with torch.no_grad():

        for i, batch in enumerate(dataloader):
            x = batch[0] if not column else batch[0].reshape(-1, 1)

            # Forward pass
            outputs = model(x)  # MTL training BTSP
            loss += criterion(outputs, x)

    model.train()

    return loss / len(dataloader), model


def testing_mod(data: np.ndarray, model: object,
                   alpha_samples: np.ndarray,
                   alpha_baseline: float=0.1,
                   criterion: object=MSELoss(),
                   column: bool=False,
                   use_tensor: bool=False,
                   progressive_test: bool=False):

    """
    Test the model

    Parameters
    ----------
    data: np.ndarray
        z data
    model: nn.Module
        the model
    """

    if not isinstance(data, DataLoader):
        # Convert numpy array to torch tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # Create a dataset and data loader
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    else:
        dataloader = data

    if use_tensor:
        try:
            data_tensor
        except NameError:
            raise ValueError("data_tensor is not defined")
        dataloader = data_tensor.unsqueeze(1)

    # Set the model to evaluation mode
    model.eval()
    loss = 0.
    acc_matrix = torch.zeros(len(dataloader), len(dataloader))

    alpha = model._alpha
    logger("testing...")

    with torch.no_grad():

        for i, batch in utils.tqdm_enumerate(dataloader):
            x = batch[0] if not column else batch[0].reshape(-1, 1)

            new_alpha = np.maximum(alpha_baseline, alpha * alpha_samples[i])

            # Forward pass
            model.set_alpha(alpha=new_alpha)
            outputs = model(x)  # MTL training BTSP
            loss += criterion(outputs, x)

    model.train()

    return loss / len(dataloader), model


def progressive_testing(data: np.ndarray, model: object):

    """
    Test the model

    Parameters
    ----------
    data: np.ndarray
        z data
    model: nn.Module
        the model
    """

    datasets = []
    for k in range(len(data)):
        data = torch.tensor(data[:k+1], dtype=torch.float32, requires_grad=False)
        dataloader = DataLoader(TensorDataset(data),
                                batch_size=1,
                                shuffle=False)
        datasets += [dataloader]

    # Set the model to evaluation mode
    model.eval()
    acc_matrix = torch.zeros(len(data),
                             len(data))

    with torch.no_grad():

        # loop over all patterns
        for i, x in enumerate(datasets[-1]):

            logger.debug(f"dataset {i}, {x}")
            x = x[0].reshape(-1, 1)

            # Forward pass
            outputs = model(x)

            # test all previous patterns
            model.pause_lr()

            # testing over all previous patterns 0.. i
            for j, x in enumerate(datasets[i]):
                x = x[0].reshape(-1, 1)

                # forward
                y = model(x)

                # record : cosine similarity
                value = (y.T @ x) / \
                    (torch.norm(x) * torch.norm(y))
                acc_matrix[i, j] = (value.item() - 0.2) / 0.8

                # logger(f"pattern {j}")

            model.resume_lr()

    model.train()

    return acc_matrix, model


def reconstruction_loss(y: np.ndarray, y_pred: np.ndarray) -> float:

    """
    Calculate the reconstruction loss

    Parameters
    ----------
    y: np.ndarray
        the original data
    y_pred: np.ndarray
        the predicted data

    Returns
    -------
    float
        the reconstruction loss
    """

    return np.mean((y - y_pred)**2)


def train_for_accuracy(alpha: float,
                       num_rep: int,
                       num_samples: int,
                       complete_dataset: object=None,
                       **kwargs) -> np.ndarray:

    """
    trainings for a given alpha (already set in the model)

    Parameters
    ----------
    alpha : float
        learning rate
    num_rep : int
        number of repetitions
    num_samples : int
        number of samples
    complete_dataset : object
        data. Default None
    **kwargs
        idx : int
            index of the autoencoder to load.
            Default 0.
        use_bias : bool
            use bias. Default True.
        shuffled : bool
            shuffled the IS. Default False.
        verbose : bool
            verbose. Default False.

    Returns
    -------
    np.ndarray
        outputs
    """

    verbose = kwargs.get("verbose", False)

    # --- load autoencoder
    logger.debug(f"{kwargs.get('idx', 0)=}")
    info, autoencoder = models.load_session(
        idx=kwargs.get("idx", 0))


    # information
    dim_ei = info["dim_ei"]
    dim_ca3 = info["dim_ca3"]
    dim_ca1 = info["dim_ca1"]
    dim_eo = info["dim_eo"]
    K_lat = info["K_lat"]
    beta = info["beta"]
    K = info["K"]

    # number training samples used for the AE
    # num_samples = info["num_samples"]

    # get parameters
    W_ei_ca1, W_ca1_eo, B_ei_ca1, B_ca1_eo = autoencoder.get_weights(
                                bias=kwargs.get("use_bias", True))

    if verbose:
        logger(f"{autoencoder=}")
        logger("<<< Loaded session >>>")

    # --- make model

    # get weights from the autoencoder
    if kwargs.get("use_bias", True):
        W_ei_ca1, W_ca1_eo, B_ei_ca1, B_ca1_eo = autoencoder.get_weights(
                                                        bias=True)
    else:
        W_ei_ca1, W_ca1_eo = autoencoder.get_weights(bias=False)
        B_ei_ca1 = None
        B_ca1_eo = None

    # make model
    model = models.MTL(W_ei_ca1=W_ei_ca1,
                W_ca1_eo=W_ca1_eo,
                B_ei_ca1=B_ei_ca1,
                B_ca1_eo=B_ca1_eo,
                dim_ca3=dim_ca3,
                K_lat=K_lat,
                K_out=K,
                alpha=alpha,
                beta=beta,
                random_IS=kwargs.get("shuffled", False))

    if verbose:
        logger(f"%MTL: {model}")

    #
    outputs = np.zeros((num_rep, num_samples, num_samples))

    if complete_dataset is None:
        complete_dataset = []
        is_dataset = False
    else:
        is_dataset = True

    for l in tqdm(range(num_rep)):

        # --- make new data
        if is_dataset:
            datasets = complete_dataset[l]
        else:
            stimuli = sparse_stimulus_generator(N=num_samples,
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
            complete_dataset += [datasets]

        # --- run new repetition
        for i in tqdm(range(num_samples), disable=True):

            # reset the model
            model.reset()

            # train a dataset with pattern index 0.. i
            model.eval()
            with torch.no_grad():

                # one pattern at a time
                for batch in datasets[i]:
                    # forward
                    _ = model(batch[-1].reshape(-1, 1))

            # --- test a dataset with pattern index 0.. i
            model.pause_lr()
            model.eval()
            with torch.no_grad():
                # one pattern at a time
                for j, batch in enumerate(datasets[i]):
                    x = batch[-1].reshape(-1, 1)

                    # forward
                    y = model(x)

                    # record : cosine similarity
                    value = (y.T @ x) / \
                        (torch.norm(x) * torch.norm(y))

                    outputs[l, i, j] = (value.item() - 0.2) / 0.8

    return outputs, model, complete_dataset


def train_for_accuracy_lec(num_rep: int,
                       num_samples: int,
                       complete_dataset: object=None,
                       alpha: float=None,
                       **kwargs) -> np.ndarray:

    """
    trainings for a given alpha (already set in the model)

    Parameters
    ----------
    alpha : float
        learning rate
    num_rep : int
        number of repetitions
    num_samples : int
        number of samples
    complete_dataset : object
        data. Default None
    **kwargs
        idx : int
            index of the autoencoder to load.
            Default 0.
        use_bias : bool
            use bias. Default True.
        shuffled : bool
            shuffled the IS. Default False.
        verbose : bool
            verbose. Default False.
        binarize : bool
            binarize the activity. Default False.

    Returns
    -------
    np.ndarray
        outputs
    """

    verbose = kwargs.get("verbose", False)

    # --- load autoencoder
    info, autoencoder = models.load_session(
        idx=kwargs.get("idx", 0), verbose=verbose)

    # information
    if "network_params" not in info:
        raise ValueError("network_params not found in the info file")

    network_params = info["network_params"]

    # get parameters
    W_ei_ca1, W_ca1_eo, B_ei_ca1, B_ca1_eo = autoencoder.get_weights(
                                bias=kwargs.get("use_bias", True))

    if verbose:
        logger(f"{autoencoder=}")
        logger("<<< Loaded session >>>")

    # --- make model

    # get weights from the autoencoder
    if kwargs.get("use_bias", True):
        W_ei_ca1, W_ca1_eo, B_ei_ca1, B_ca1_eo = autoencoder.get_weights(bias=True)
    else:
        W_ei_ca1, W_ca1_eo = autoencoder.get_weights(bias=False)
        B_ei_ca1 = None
        B_ca1_eo = None

    # make model
    model = models.MTL(W_ei_ca1=W_ei_ca1,
            W_ca1_eo=W_ca1_eo,
            B_ei_ca1=B_ei_ca1,
            B_ca1_eo=B_ca1_eo,
            dim_ca3=network_params["dim_ca3"],
            K_lat=network_params["K_ca1"],
            K_out=network_params["K_eo"],
            K_ca3=network_params["K_ca3"],
            beta=network_params["beta_ca1"],
            alpha=network_params["alpha"] if alpha is None else alpha,
            identity_IS=False,
            random_IS=kwargs.get("shuffled", False))

    if verbose:
        logger(f"%MTL: {model}")

    #
    outputs = np.zeros((num_rep, num_samples, num_samples))

    if complete_dataset is None:
        complete_dataset = []
        is_dataset = False
    else:
        is_dataset = True

    for l in tqdm(range(num_rep)):

        # --- make new data
        if is_dataset:
            datasets = complete_dataset[l]
        else:
            stimuli, _, _ = sparse_stimulus_generator_sensory(
                                num_stimuli=num_samples,
                                K = network_params["K_lec"],
                                mec_size=network_params["dim_mec"],
                                lec_size=network_params["dim_lec"],
                                N_x=network_params["mec_N_x"],
                                N_y=network_params["mec_N_y"],
                                pf_sigma=network_params["mec_sigma"],
                                lap_length=network_params["mec_N_x"],
                                num_laps=None,
                                num_cues=None,
                                position_list=None,
                                cue_positions=None,
                                sen_list=None,
                                plot=False,
                                binarize=kwargs.get("binarize", False))
            datasets = []
            for k in range(num_samples):
                data = torch.tensor(stimuli[:k+1], dtype=torch.float32)
                dataloader = DataLoader(TensorDataset(data),
                                        batch_size=1,
                                        shuffle=False)
                datasets += [dataloader]
            complete_dataset += [datasets]

        # --- run new repetition
        for i in tqdm(range(num_samples), disable=True):

            # reset the model
            model.reset()

            # train a dataset with pattern index 0.. i
            model.eval()
            with torch.no_grad():

                # one pattern at a time
                for batch in datasets[i]:
                    # forward
                    _ = model(batch[-1].reshape(-1, 1))

            # --- test a dataset with pattern index 0.. i
            model.pause_lr()
            model.eval()
            with torch.no_grad():
                # one pattern at a time
                for j, batch in enumerate(datasets[i]):
                    x = batch[-1].reshape(-1, 1)

                    # forward
                    y = model(x)

                    # record : cosine similarity
                    value = (y.T @ x) / \
                        (torch.norm(x) * torch.norm(y))

                    outputs[l, i, j] = (value.item() - 0.2) / 0.8

    return outputs, model, complete_dataset


def train_for_accuracy_v2(alpha: float,
                       num_rep: int,
                       num_samples: int,
                       complete_dataset: object=None,
                       **kwargs) -> np.ndarray:

    """
    trainings for a given alpha (already set in the model)

    Parameters
    ----------
    alpha : float
        learning rate
    num_rep : int
        number of repetitions
    num_samples : int
        number of samples
    complete_dataset : object
        data. Default None
    **kwargs
        idx : int
            index of the autoencoder to load.
            Default 0.
        use_bias : bool
            use bias. Default True.
        shuffled : bool
            shuffled the IS. Default False.
        verbose : bool
            verbose. Default False.

    Returns
    -------
    np.ndarray
        outputs
    """

    verbose = kwargs.get("verbose", False)

    # --- load autoencoder
    logger.debug(f"{kwargs.get('idx', 0)=}")
    info, autoencoder = models.load_session(
        idx=kwargs.get("idx", 0))


    # information
    dim_ei = info["dim_ei"]
    dim_ca3 = info["dim_ca3"]
    dim_ca1 = info["dim_ca1"]
    dim_eo = info["dim_eo"]
    K_lat = info["K_lat"]
    beta = info["beta"]
    K = info["K"]

    # number training samples used for the AE
    # num_samples = info["num_samples"]

    # get parameters
    W_ei_ca1, W_ca1_eo, B_ei_ca1, B_ca1_eo = autoencoder.get_weights(
                                bias=kwargs.get("use_bias", True))

    if verbose:
        logger(f"{autoencoder=}")
        logger("<<< Loaded session >>>")

    # --- make model

    # get weights from the autoencoder
    if kwargs.get("use_bias", True):
        W_ei_ca1, W_ca1_eo, B_ei_ca1, B_ca1_eo = autoencoder.get_weights(
                                                        bias=True)
    else:
        W_ei_ca1, W_ca1_eo = autoencoder.get_weights(bias=False)
        B_ei_ca1 = None
        B_ca1_eo = None

    # make model
    model = models.MTL(W_ei_ca1=W_ei_ca1,
                W_ca1_eo=W_ca1_eo,
                B_ei_ca1=B_ei_ca1,
                B_ca1_eo=B_ca1_eo,
                dim_ca3=dim_ca3,
                K_lat=K_lat,
                K_out=K,
                alpha=alpha,
                beta=beta,
                random_IS=kwargs.get("shuffled", False))

    if verbose:
        logger(f"%MTL: {model}")

    #
    outputs = np.zeros((num_rep, num_samples, num_samples))
    activity = []

    if complete_dataset is None:
        complete_dataset = []
        is_dataset = False
    else:
        is_dataset = True

    for l in tqdm(range(num_rep)):

        # --- make new data
        if is_dataset:
            datasets = complete_dataset[l]
        else:
            stimuli = sparse_stimulus_generator(N=num_samples,
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
            complete_dataset += [datasets]

        # --- run new repetition
        for i in tqdm(range(num_samples), disable=True):

            episode_logs = []

            # reset the model
            model.reset()

            # train a dataset with pattern index 0.. i
            model.eval()
            with torch.no_grad():

                # one pattern at a time
                for batch in datasets[i]:
                    # forward
                    x = batch[-1].reshape(-1, 1)
                    y = model(x)
                    loss = cross_entropy(x.reshape(-1), y.reshape(-1))

                    logs = [x.reshape(-1).detach().numpy(), y.detach().numpy(), loss.item()]

                    y = model(x, test=True)
                    loss = cross_entropy(x.reshape(-1), y.reshape(-1))
                    logs += [y.detach().numpy(), loss.item()]

                    episode_logs += [logs]

            # --- test a dataset with pattern index 0.. i
            model.pause_lr()
            model.eval()
            with torch.no_grad():
                # one pattern at a time
                for j, batch in enumerate(datasets[i]):
                    x = batch[-1].reshape(-1, 1)

                    # forward
                    y = model(x)

                    # record : cosine similarity
                    loss = cross_entropy(x.reshape(-1), y.reshape(-1))
                    norm_ = cosine_similarity_vec(x, y)

                    outputs[l, i, j] = (norm_.item() - 0.2) / 0.8

                    episode_logs[j] += [x.reshape(-1).detach().numpy(), y.detach().numpy(), loss.item()]
                    activity += [episode_logs[j]]

    return outputs, model, complete_dataset, activity


def train_for_reconstruction(alpha: float,
                             num_samples: int,
                             use_lec: bool=False,
                             **kwargs) -> np.ndarray:

    """
    trainings for a given alpha (already set in the model)

    Parameters
    ----------
    alpha : float
        learning rate
    num_samples : int
        number of samples
    use_lec: bool
        use LEC. Default True.
    **kwargs
        idx : int
            index of the autoencoder to load.
            Default 0.
        use_bias : bool
            use bias. Default True.
        binarize : bool
            binarize the activity. Default False.

    Returns
    -------
    np.ndarray
        outputs
    """

    if use_lec:
        logger("using LEC data")
        _, model, complete_dataset = train_for_accuracy_lec(alpha=alpha,
                            num_rep=1,
                            num_samples=num_samples,
                            idx=kwargs.get("idx", 0),
                            use_bias=kwargs.get("use_bias", True),
                            binarize=kwargs.get("binarize", False))

        _, model_rnd, _ = train_for_accuracy_lec(alpha=alpha,
                            num_rep=1,
                            num_samples=num_samples,
                            idx=kwargs.get("idx", 0),
                            use_bias=kwargs.get("use_bias", True),
                            shuffled=True,
                            complete_dataset=complete_dataset,
                            binarize=kwargs.get("binarize", False))
    else:
        _, model, complete_dataset = train_for_accuracy(alpha=alpha,
                            num_rep=1,
                            num_samples=num_samples,
                            idx=kwargs.get("idx", 0),
                            use_bias=kwargs.get("use_bias", True))

        _, model_rnd, _ = train_for_accuracy(alpha=alpha,
                            num_rep=1,
                            num_samples=num_samples,
                            idx=kwargs.get("idx", 0),
                            use_bias=kwargs.get("use_bias", True),
                            shuffled=True,
                            complete_dataset=complete_dataset)

    data = complete_dataset[-1][-1].dataset.tensors[0]
    num = len(data)

    # --- reconstruct data
    #
    model.pause_lr()
    out_mtl, latent_mtl = reconstruct_data(
                     data=data,
                     num=num,
                     model=model,
                     column=True,
                     plot=False)
    rec_loss = np.mean((data.numpy() - out_mtl)**2).item()

    #
    model_rnd.pause_lr()
    out_mtl_rnd, latent_mtl_rnd = reconstruct_data(
                     data=data,
                     num=num,
                     model=model_rnd,
                     column=True,
                     plot=False)
    rec_loss_rnd = np.mean(
        (data.numpy() - out_mtl_rnd)**2).item()

    # --- load autoencoder
    _, autoencoder = models.load_session(
        idx=kwargs.get("idx", 0), verbose=False)

    logger.debug(f"{len(data)=}, {len(data[0])=}")
    logger.debug(f"{data.shape=}")
    out_mtl_ae, latent_mtl_ae = reconstruct_data(
                     data=data,
                     num=num,
                     model=autoencoder,
                     column=False,
                     plot=False)
    rec_loss_rnd = np.mean(
        (data.numpy() - out_mtl_rnd)**2).item()

    record = {
        "data": data.numpy(),
        "out_mtl": out_mtl,
        "out_mtl_rnd": out_mtl_rnd,
        "out_ae": out_mtl_ae,
        "rec_loss": rec_loss,
        "rec_loss_rnd": rec_loss_rnd
    }

    return record


def train_for_reconstruction_v2(alpha: float,
                                num_samples: int,
                                use_lec: bool=False,
                                **kwargs) -> np.ndarray:

    """
    trainings for a given alpha (already set in the model)

    Parameters
    ----------
    alpha : float
        learning rate
    num_samples : int
        number of samples
    use_lec: bool
        use LEC. Default True.
    **kwargs
        idx : int
            index of the autoencoder to load.
            Default 0.
        use_bias : bool
            use bias. Default True.
        binarize : bool
            binarize the activity. Default False.

    Returns
    -------
    np.ndarray
        outputs
    """

    if use_lec:
        logger("using LEC data")
        _, model, complete_dataset = train_for_accuracy_lec(alpha=alpha,
                            num_rep=1,
                            num_samples=num_samples,
                            idx=kwargs.get("idx", 0),
                            use_bias=kwargs.get("use_bias", True),
                            binarize=kwargs.get("binarize", False))

        _, model_rnd, _ = train_for_accuracy_lec(alpha=alpha,
                            num_rep=1,
                            num_samples=num_samples,
                            idx=kwargs.get("idx", 0),
                            use_bias=kwargs.get("use_bias", True),
                            shuffled=True,
                            complete_dataset=complete_dataset,
                            binarize=kwargs.get("binarize", False))
    else:
        _, model, complete_dataset, activity_mtl = train_for_accuracy_v2(alpha=alpha,
                            num_rep=1,
                            num_samples=num_samples,
                            idx=kwargs.get("idx", 0),
                            use_bias=kwargs.get("use_bias", True))

        _, model_rnd, _, activity_rnd = train_for_accuracy_v2(alpha=alpha,
                            num_rep=1,
                            num_samples=num_samples,
                            idx=kwargs.get("idx", 0),
                            use_bias=kwargs.get("use_bias", True),
                            shuffled=True,
                            complete_dataset=complete_dataset)

    data = complete_dataset[-1][-1].dataset.tensors[0]
    num = len(data)

    # --- reconstruct data
    #
    model.pause_lr()
    out_mtl, latent_mtl = reconstruct_data(
                     data=data,
                     num=num,
                     model=model,
                     column=True,
                     plot=False)
    rec_loss = np.mean((data.numpy() - out_mtl)**2).item()

    #
    model_rnd.pause_lr()
    out_mtl_rnd, latent_mtl_rnd = reconstruct_data(
                     data=data,
                     num=num,
                     model=model_rnd,
                     column=True,
                     plot=False)
    rec_loss_rnd = np.mean(
        (data.numpy() - out_mtl_rnd)**2).item()

    # --- load autoencoder
    _, autoencoder = models.load_session(
        idx=kwargs.get("idx", 0), verbose=False)

    logger.debug(f"{len(data)=}, {len(data[0])=}")
    logger.debug(f"{data.shape=}")
    out_mtl_ae, latent_mtl_ae = reconstruct_data(
                     data=data,
                     num=num,
                     model=autoencoder,
                     column=False,
                     plot=False)
    rec_loss_rnd = np.mean(
        (data.numpy() - out_mtl_rnd)**2).item()

    record = {
        "data": data.numpy(),
        "out_mtl": out_mtl,
        "out_mtl_rnd": out_mtl_rnd,
        "out_ae": out_mtl_ae,
        "rec_loss": rec_loss,
        "rec_loss_rnd": rec_loss_rnd,
        "activity_mtl": activity_mtl,
        "activity_rnd": activity_rnd
    }

    return record


def train_for_weight_plot(alpha: float,
                       num_rep: int,
                       num_samples: int,
                       complete_dataset: object=None,
                       **kwargs) -> np.ndarray:

    """
    trainings for a given alpha (already set in the model)

    Parameters
    ----------
    alpha : float
        learning rate
    num_rep : int
        number of repetitions
    num_samples : int
        number of samples
    complete_dataset : object
        data. Default None
    **kwargs
        idx : int
            index of the autoencoder to load.
            Default 0.
        use_bias : bool
            use bias. Default True.
        shuffled : bool
            shuffled the IS. Default False.
        verbose : bool
            verbose. Default False.

    Returns
    -------
    np.ndarray
        outputs
    """

    verbose = kwargs.get("verbose", False)

    # --- load autoencoder
    info, autoencoder = models.load_session(
        idx=kwargs.get("idx", 0), verbose=verbose)

    # information
    dim_ei = info["dim_ei"]
    dim_ca3 = info["dim_ca3"]
    dim_ca1 = info["dim_ca1"]
    dim_eo = info["dim_eo"]
    K_lat = info["K_lat"]
    beta = info["beta"]
    K = info["K"]

    # number training samples used for the AE
    # num_samples = info["num_samples"]

    # get parameters
    W_ei_ca1, W_ca1_eo, B_ei_ca1, B_ca1_eo = autoencoder.get_weights(
                                bias=kwargs.get("use_bias", True))

    if verbose:
        logger(f"{autoencoder=}")
        logger("<<< Loaded session >>>")

    # --- make model

    # get weights from the autoencoder
    if kwargs.get("use_bias", True):
        W_ei_ca1, W_ca1_eo, B_ei_ca1, B_ca1_eo = autoencoder.get_weights(bias=True)
    else:
        W_ei_ca1, W_ca1_eo = autoencoder.get_weights(bias=False)
        B_ei_ca1 = None
        B_ca1_eo = None

    # make model
    model = models.MTL(W_ei_ca1=W_ei_ca1,
                W_ca1_eo=W_ca1_eo,
                B_ei_ca1=B_ei_ca1,
                B_ca1_eo=B_ca1_eo,
                dim_ca3=dim_ca3,
                K_lat=K_lat,
                K_out=K,
                alpha=alpha,
                beta=beta,
                shuffled_is=kwargs.get("shuffled", False))

    if verbose:
        logger(f"%MTL: {model}")

    #
    record = np.zeros((num_rep, num_samples, dim_ca3, dim_ca3))

    if complete_dataset is None:
        complete_dataset = []
        is_dataset = False
    else:
        is_dataset = True

    for l in tqdm(range(num_rep)):

        # --- make new data
        if is_dataset:
            datasets = complete_dataset[l]
        else:
            stimuli = sparse_stimulus_generator(N=num_samples,
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
            complete_dataset += [datasets]

        # --- run new repetition
        for i in tqdm(range(num_samples), disable=True):

            # reset the model
            model.reset()

            # train a dataset with pattern index 0.. i
            model.eval()
            with torch.no_grad():

                # one pattern at a time
                for batch in datasets[i]:

                    # forward
                    _ = model(batch[0].reshape(-1, 1))


    return outputs, model, complete_dataset


def testing_mod(data: np.ndarray, model: object,
                   alpha_samples: np.ndarray,
                   alpha_baseline: float=0.1,
                   criterion: object=MSELoss(),
                   column: bool=False,
                   use_tensor: bool=False,
                   progressive_test: bool=False):

    """
    Test the model

    Parameters
    ----------
    data: np.ndarray
        z data
    model: nn.Module
        the model
    """

    if not isinstance(data, DataLoader):
        # Convert numpy array to torch tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # Create a dataset and data loader
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    else:
        dataloader = data

    if use_tensor:
        try:
            data_tensor
        except NameError:
            raise ValueError("data_tensor is not defined")
        dataloader = data_tensor.unsqueeze(1)

    # Set the model to evaluation mode
    model.eval()
    loss = 0.
    acc_matrix = torch.zeros(len(dataloader), len(dataloader))

    alpha = model._alpha
    logger("testing...")

    with torch.no_grad():

        for i, batch in tqdm_enumerate(dataloader):
            x = batch[0] if not column else batch[0].reshape(-1, 1)

            new_alpha = np.maximum(alpha_baseline, alpha * alpha_samples[i])

            # Forward pass
            model.set_alpha(alpha=new_alpha)
            outputs = model(x)  # MTL training BTSP
            loss += criterion(outputs, x)

    model.train()

    return loss / len(dataloader), model


""" EXPERIMENTS """

def sparse_stimulus_generator_sensory(num_stimuli: int, K : int,
                                      mec_size: int,  lec_size: int,
                                      N_x : int, N_y : int,
                                      pf_sigma: int,
                                      num_laps: int,
                                      cue_duration: int=1,
                                      lap_length: int=None,
                                      num_cues: int=None,
                                      position_list=None,
                                      cue_positions=None,
                                      verbose: bool=False,
                                      sen_list=None,
                                      plot: bool=False,
                                      sigma: float=5) -> np.ndarray:

    """
    This function generates random z patterns with a certain
    degree of sparsity

    Parameters
    ----------
    N : int
        Number of samples
    K : int
        Number of active units
    size : int, optional
        Size of the z patterns, by default 10
    plot : bool, optional
        Whether to plot the z patterns.
        Default False

    Returns
    -------
    samples : np.ndarray
        z patterns
    """

    def place_field_activity(N_x, N_y, sigma, xi, yi):
        """
        Computes place field activity for each cell on an NxN grid for a given location (xi, yi).
        """

        def circular_distance(x1, x2, N):
            """
            Computes the minimum circular distance in the x-direction (wraps around the boundaries).
            """
            return np.minimum(np.abs(x1 - x2), N - np.abs(x1 - x2))

        # Create a grid of size NxN with place cells at each position
        x = np.linspace(0, N_x-1, N_x)
        y = np.linspace(0, N_y-1, N_y)
        X, Y = np.meshgrid(x, y)
        # Calculate the squared Euclidean distance between (xi, yi) and each place cell location
        dist_squared = circular_distance(X, xi, N_x) ** 2 + (Y - yi) ** 2

        # Compute Gaussian activity for each place cell
        activity = np.exp(-dist_squared / (2 * sigma ** 2))
        return activity

    def pc_cue_probability(x, c, sigma):
        p = np.exp(-((x - c) ** 2) / (2 * sigma ** 2))
        return np.random.binomial(1, p)

    samples = np.zeros((num_stimuli, mec_size + lec_size))
    alpha_samples = np.zeros(num_stimuli)

    #cue_pattern_size = lec_size//num_cues  #
    if lec_size > 0 and num_cues is not None:
        fixed_cue = np.zeros((num_cues, lec_size))
        for k in range(num_cues):
            cue_idx = np.random.choice(range(lec_size), replace=False, size=K)
            #fixed_cue[cue_idx] = 1
            fixed_cue[k, K*k: K*(k+1)] = 1  # | 1, 1, .0s  , 1, 1, ..0s.. |

    # 0, 0, 1, 0, 1, 1...
    #lap_cues = np.random.choice(range(num_cues), size=num_laps) if num_laps is not None else None
    lap_cues = np.zeros(num_laps) if position_list is not None else None

    position_pool = np.arange(N_x)  # Possible positions in 1D track

    """
    [0, 1, 3, 4.. 10, 0, 1, 3, ...10]
    """
    lap_i = 0
    for i in range(num_stimuli): # laps x length

        if cue_positions is not None:

            # count laps
            if i % lap_length == 0:
                lap_idx = (lap_i // cue_duration) % num_cues
                lap_cues[lap_i] = lap_idx

                if verbose:
                    print("---------------------------")
                    print(f"lap {lap_i}, cue {lap_idx}")
                lap_i += 1

        """
        if mec_size > 0:
            x_i, y_i = (position_list[i] if position_list is not None \
                                else (np.random.randint(0, N_x), np.random.randint(0, N_y)))
            activity_grid = place_field_activity(N_x, N_y, pf_sigma, x_i, y_i)
            samples[i, :mec_size] = activity_grid.flatten()

        """
        if mec_size > 0:
            # Reset the pool every 50 samples
            if position_list is None:
              if i % mec_size == 0:
                  np.random.shuffle(position_pool)  # Shuffle the pool to get random order
              pos_idx = i % 50  # Get position index within the shuffled pool
              x_i = position_pool[pos_idx]  # Choose the x position
              y_i = np.random.randint(0, N_y)
            else:
              x_i, y_i = position_list[i]
            activity_grid = place_field_activity(N_x, N_y, pf_sigma, x_i, y_i)
            samples[i, :mec_size] = activity_grid.flatten()

        if lec_size > 0:

            # cue positions provided
            if cue_positions is not None:
                p = samples[i, cue_positions[lap_idx]] / \
                    samples[i, :mec_size].max()
                alpha_samples[i] = p

                if np.random.binomial(1, p):
                    activity_lec = fixed_cue[lap_cues[lap_idx].astype(int)]
                else:
                    activity_lec = np.zeros((lec_size))
                    lec_idx = np.random.choice(range(lec_size),
                                               replace=False, size=K)
                    activity_lec[lec_idx] = 1
            else:

                if cue_positions is not None:
                    lec_idx = np.random.choice(range(lec_size),
                                               replace=False, size=K)
                    activity_lec[lec_idx] = 1

                else:
                    activity_lec = np.zeros((lec_size))
                    lec_idx = np.random.choice(range(lec_size),
                                           replace=False, size=K)
                    activity_lec[lec_idx] = 1

            samples[i, mec_size:] = sen_list[i] if sen_list is not None else activity_lec

    samples = samples.astype(np.float32) # [ 00000 |      ]

    return samples, lap_cues, alpha_samples



def get_track_input(track_params: dict, network_params: dict, cue_duration: int=1):

  position_list = [(x, 0) for lap in range(track_params["num_laps"]) for x in range(track_params["length"])]

  if track_params["reward"] == "random":
    reward_list = None
  if track_params["cue"] == "random":
    cue_list = None

  sen_list = None

  track_input, lap_cues, alpha_samples = sparse_stimulus_generator_sensory(num_stimuli=track_params["num_laps"]*track_params["length"],
                                                  K = network_params["K_lec"],
                                                  mec_size=network_params["dim_mec"],
                                                  lec_size=network_params["dim_lec"],
                                                  N_x=network_params["mec_N_x"],
                                                  N_y=network_params["mec_N_y"],
                                                  cue_duration=cue_duration,
                                                  pf_sigma=network_params["mec_sigma"],
                                                  lap_length=track_params["length"],
                                                  num_laps=track_params["num_laps"],
                                                  num_cues=network_params["num_cues"],
                                                  position_list=position_list,
                                                  cue_positions=track_params["cue_position"],
                                                  sen_list=None,
                                                  plot=False)
  return track_input, lap_cues, alpha_samples


if __name__ == "__main__":



    # # generate the z patterns
    # N = 10
    # size = 50

    # heads = 3
    # variance = 0.01

    # higher_heads = heads
    # higher_variance = 0.5

    # samples = stimulus_generator(N, size, heads, variance,
    #                              higher_heads=higher_heads,
    #                              higher_variance=higher_variance,
    #                              plot=True)


    """ test activation function """

    # z = torch.randn(6)
    # print(f"z: {z}")
    # z_soft = torch.nn.functional.softmax(z, dim=-1)
    # print(f"z_soft: {z_soft}")

    # plt.subplot(311)
    # plt.imshow(z.numpy().reshape(1, -1),
    #            aspect="auto", vmin=-2, vmax=2)
    # plt.title("input")
    # plt.axhline(0, color="black", alpha=0.2)
    # plt.plot(z.numpy(), label="z", alpha=0.4)

    # softsigmoid = SoftSigmoid(gamma=2.,
    #                           beta=1.,
    #                           alpha=0.5)
    # z_sigmoid = softsigmoid(z)

    # print(f"[1] z_sigmoid: {z_sigmoid}")
    # plt.subplot(312)
    # plt.imshow(z_sigmoid.numpy().reshape(1, -1),
    #            aspect="auto", vmin=-2, vmax=2)
    # plt.plot(z_sigmoid.numpy(), label="1")

    # softsigmoid = SoftSigmoid(gamma=2.,
    #                           beta=100.,
    #                           alpha=0.2)
    # z_sigmoid = softsigmoid(z)

    # print(f"[2] z_sigmoid: {z_sigmoid}")
    # plt.subplot(313)
    # plt.imshow(z_sigmoid.numpy().reshape(1, -1),
    #            aspect="auto", vmin=-2, vmax=2)
    # plt.plot(z_sigmoid.numpy(), label="2")

    # plt.ylim(-2, 2)
    # plt.legend()
    # plt.grid()
    # plt.show()

    # --- test spars stimulus generator ---
    N = 2
    data = sparse_stimulus_generator(N, K=5, size=50, plot=True)





