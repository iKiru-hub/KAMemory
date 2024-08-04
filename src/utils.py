import numpy as np
import matplotlib.pyplot as plt
import warnings

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.autograd as autograd

import logging, coloredlogs
from tqdm import tqdm



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
    data_tensor = torch.tensor(data[:num],
                               dtype=torch.float32)

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
            column: bool=False):

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

    # Set the model to evaluation mode
    model.eval()

    loss = 0.

    with torch.no_grad():

        for batch in dataloader:

            zs = batch[0] if not column else batch[0].reshape(-1, 1)

            # Forward pass
            outputs = model(zs)

            # evaluate the output
            loss += criterion(outputs, zs)

    model.train()

    return loss / len(dataloader), model




""" miscellanous """

# logger
def setup_logger(name: str="MAIN", colored: bool=True) -> logging.Logger:

    """
    this function sets up a logger

    Parameters
    ----------
    name : str
        name of the logger. Default="MAIN"
    colored : bool
        use colored logs. Default=True

    Returns
    -------
    logger : object
        logger object
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create a custom formatter
    if colored:
        formatter = coloredlogs.ColoredFormatter(
            "%(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # create a colored stream handler with the custom formatter
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        # add the handler to the logger and disable propagation
        logger.addHandler(handler)

    logger.propagate = False

    # wrapper class 
    class LoggerWrapper:
        def __init__(self, logger):
            self.logger = logger

        def __repr__(self):

            return f"LoggerWrapper(name={self.logger.name})"

        def __call__(self, msg: str=""):
            self.logger.info(msg)

        def info(self, msg):
            self.logger.info(msg)

        def warning(self, msg):
            self.logger.warning(msg)

        def error(self, msg):
            self.logger.error(msg)

        def debug(self, msg, DEBUG: bool=True):
            if DEBUG:
                self.logger.debug(msg)

    return LoggerWrapper(logger)


def plot_squashed_data(data: np.ndarray, title: str="",
                       ax: plt.Axes=None, squash: bool=False):

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
    ax.set_ylabel(title)
    ax.set_yticks(range(len(data)))
    ax.set_xticks([])

    if ax is None:
        plt.show()


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

        print(gamma)
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

    z = torch.randn(6)
    print(f"z: {z}")
    z_soft = torch.nn.functional.softmax(z, dim=-1)
    print(f"z_soft: {z_soft}")
    # plt.subplot(311)
    # plt.imshow(z.numpy().reshape(1, -1), aspect="auto", vmin=-2, vmax=2)
    plt.axhline(0, color="black", alpha=0.2)
    plt.plot(z.numpy(), label="z", alpha=0.4)

    softsigmoid = SoftSigmoid(gamma=2.,
                              beta=1.,
                              alpha=0.5)
    z_sigmoid = softsigmoid(z)

    print(f"[1] z_sigmoid: {z_sigmoid}")
    # plt.subplot(312)
    # plt.imshow(z_sigmoid.numpy().reshape(1, -1), aspect="auto", vmin=-2, vmax=2)
    plt.plot(z_sigmoid.numpy(), label="1")

    softsigmoid = SoftSigmoid(gamma=2.,
                              beta=100.,
                              alpha=0.2)
    z_sigmoid = softsigmoid(z)

    print(f"[2] z_sigmoid: {z_sigmoid}")
    # plt.imshow(z_sigmoid.numpy().reshape(1, -1), aspect="auto", vmin=-2, vmax=2)
    # plt.plot(z_sigmoid.numpy(), label="2")

    # plt.ylim(-2, 2)
    # plt.legend()
    # plt.grid()
    # plt.show()

    # --- test spars stimulus generator ---
    N = 10
    data = sparse_stimulus_generator(N, K=5, size=50, plot=True)


