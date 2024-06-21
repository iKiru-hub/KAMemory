import numpy as np
import matplotlib.pyplot as plt
import warnings



def stimulus_generator(N: int, size: int=10, heads: int=2, variance: float=0.1,
                       higher_heads: int=None, higher_variance: float=None,
                       plot: bool=False) -> np.ndarray:

    """
    This function generates random input patterns with a certain
    degree of structure

    Parameters
    ----------
    N : int
        Number of samples
    size : int, optional
        Size of the input patterns, by default 10
    # generate docstring
        heads : int, optional
        Number of heads, by default 2
    variance : float, optional
        Variance of the Gaussian used to generate the input patterns, by default 0.1
    higher_heads : int, optional
        Higher number of heads, by default None
    higher_variance : float, optional
        Higher variance of the Gaussian used to generate the input patterns, by default None
    plot : bool, optional
        Whether to plot the input patterns, by default False

    Returns 
    -------
    samples : np.ndarray
        Input patterns
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
            for k, (hh, hv) in enumerate(zip(high_mu, high_variance)):
                mu[i, k] = np.random.normal(hh, hv)
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

    # generate the input patterns
    samples = np.zeros((N, size))
    for i in range(N):
        for k in range(heads):
            for x in range(size):
                p = np.exp(-((x-mu[i, k])**2)/(2*variance[i, k]))
                samples[i, x] += np.random.binomial(1, p)

    if plot:
        plot_stimuli(samples=samples)

    return samples


def plot_stimuli(samples: np.ndarray):

    """
    This function plots the input patterns

    Parameters
    ----------
    samples : np.ndarray
        Input patterns
    """

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    ax.imshow(samples, aspect="auto", vmin=0, vmax=1, cmap="gray_r")
    ax.set_xlabel("size")
    ax.set_ylabel("samples")
    if len(samples) < 20:
        ax.set_yticks(range(samples.shape[0]))
        ax.set_yticklabels(range(1, 1+samples.shape[0]))
    ax.set_title("Input patterns")

    ax2.imshow(samples.sum(axis=0).reshape(1, -1),
               aspect="auto", cmap="gray_r")
    ax2.set_xlabel("size")
    ax2.set_title("Average input pattern")
    plt.show()


if __name__ == "__main__":

    # generate the input patterns
    N = 500
    size = 100

    # sampling a sensory experience from
    # an episode
    heads = 2
    variance = 0.01  # noise within-episodes

    # sampling an episode from the true world model
    higher_heads = heads
    higher_variance = 0.3  # noise among-episodes

    samples = stimulus_generator(N, size, heads, variance,
                                 higher_heads=higher_heads,
                                 higher_variance=higher_variance,
                                 plot=True)

