import torch
from torch import nn

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, json, sys
from pprint import pprint

sys.path.append(os.path.join(os.getcwd().split("KAMemory")[0], "KAMemory/src"))

# local
import utils
from logger import logger

# cache_dir = "cache"
cache_dir = os.getcwd().split("KAMemory")[0] + "KAMemory/cache"
cache_dir_2 = os.getcwd().split("KAMemory")[0] + "KAMemory/src/cache"


""" Autoencoder """

class Autoencoder(nn.Module):

    def __init__(self, input_dim: int=10, encoding_dim=10,
                 activation: str=None, K: int=10, beta: float=20.,
                 use_bias: bool=True):

        """
        Simple autoencoder with a single linear layer as encoder and decoder.

        Parameters
        ----------
        input_dim: int
            the size of the input data
        encoding_dim: int
            the size of the encoded data
        K: int
            the number of top values to select.
            Default is 10
        beta: float
            the beta value for the sparsemoid function.
            Default is 20.
        use_bias: bool
            use bias in the linear layers.
            Default is True.
        """

        super(Autoencoder, self).__init__()

        self._input_dim = input_dim
        self._encoding_dim = encoding_dim
        self._K = K
        self._beta = beta
        self._use_bias = use_bias

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim, bias=use_bias),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim, bias=use_bias),
        )

    def forward(self, x: torch.Tensor, ca1: bool=False):

        """
        Forward pass

        Parameters
        ----------
        x: torch.Tensor
            input data
        ca1: bool
            return the data from CA1. Default is False

        Returns
        -------
        torch.Tensor
            reconstructed data
        """

        z = self.encoder(x)
        z = utils.sparsemoid(z=z, K=self._K,
                             beta=self._beta)

        # ---
        x = self.decoder(z)
        x = torch.sigmoid(10*(x-0.1))

        if ca1:
            return x, z

        return x

    def get_weights(self, bias: bool=False):

        """
        Get the weights of the autoencoder model

        Returns
        -------
        tuple
            the weights of the encoder and decoder
        """

        ei_ca1 = self.encoder[0].weight.data.reshape(self._encoding_dim, self._input_dim)
        ca1_eo = self.decoder[0].weight.data.reshape(self._input_dim, self._encoding_dim)

        if bias and self._use_bias:
            ei_ca1_b = self.encoder[0].bias.data.reshape(-1, 1)
            ca1_eo_b = self.decoder[0].bias.data.reshape(-1, 1)

            return ei_ca1, ca1_eo, ei_ca1_b, ca1_eo_b

        return ei_ca1, ca1_eo, None, None


""" Main model """


Kis = 50

class MTL(nn.Module):

    def __init__(self, W_ei_ca1: torch.Tensor,
                 W_ca1_eo: torch.Tensor,
                 K_lat: int,
                 K_out: int,
                 dim_ca3: int,
                 beta: float,
                 alpha: float=0.01,
                 K_ca3: int=10,
                 identity_IS : bool=False,
                 random_IS : bool=False,
                 B_ei_ca1: torch.Tensor=None,
                 B_ca1_eo: torch.Tensor=None):

        # make docstrings
        """
        Multi-target learning model with BTSP learning rule

        Parameters
        ----------
        W_ei_ca1: torch.Tensor
            the weight matrix from entorhinal cortex to CA1
        W_ca1_eo: torch.Tensor
            the weight matrix from CA1 to entorhinal cortex output
        B_ei_ca1: torch.Tensor
            the bias for the EC to CA1 layer.
            Default is None
        B_ca1_eo: torch.Tensor
            the bias for the CA1 to EC output layer.
            Default is None
        K_lat: int
            the number of top values to select
        K_out: int
            the number of top values to select for the output
        K_ca3: int
            the number of top values to select for the CA3 layer
        beta: float
            the beta value for the sparsemoid function
        alpha: float
            the learning rate for the weight update
        dim_ca3: int
            the size of the CA3 layer
        """

        super(MTL, self).__init__()

        # infer dimensions of EC input and output and CA1
        self._dim_ei = W_ei_ca1.shape[1]
        self._dim_eo = W_ca1_eo.shape[0]
        self._dim_ca1 = W_ca1_eo.shape[1]

        # network parameters
        self._K_lat = K_lat
        self._K_ca3 = K_ca3
        self._K_out = K_out
        self._beta = beta
        self._beta_ca3 = 100*beta
        self._alpha = alpha

        # Initialize weight matrices for each layer
        self.W_ei_ca3 = nn.Parameter(torch.randn(dim_ca3,
                                                 self._dim_ei) / dim_ca3)
        self.W_ei_ca1 = nn.Parameter(W_ei_ca1)
        self.W_ca3_ca1 = nn.Parameter(torch.zeros(self._dim_ca1, dim_ca3))
        self.W_ca1_eo = nn.Parameter(W_ca1_eo)

        self.B_ei_ca1 = nn.Parameter(torch.zeros(self._dim_ca1, 1) \
                                    if B_ei_ca1 is None else B_ei_ca1)
        self.B_ca1_eo = nn.Parameter(torch.zeros(self._dim_eo, 1) \
                                    if B_ca1_eo is None else B_ca1_eo)
        self.is_bias = B_ei_ca1 is not None and B_ca1_eo is not None

        self._ca1 = None
        self._ca3 = None
        self._eo = None

        self.identity_IS = identity_IS
        self.random_IS = random_IS

        # mode
        self.mode = "train"

        self.recordings = {}
        self.recordings["x_ei"] = []
        self.recordings["ca3"] = []
        self.recordings["IS"] = []
        self.recordings["ca1"] = []
        self.recordings["eo"] = []
        self.recordings["W_ca3_ca1"] = []

    def __repr__(self):

        return f"MTL(dim_ei={self._dim_ei}, dim_ca1={self._dim_ca1}," + \
            f" dim_ca3={self.W_ei_ca3.shape[0]}, dim_eo={self._dim_eo}, " + \
            f" bias={self.is_bias}, " + \
            f"beta={self._beta}, alpha={self._alpha}, K_l={self._K_lat}, " + \
            f"K_o={self._K_out}"

    def forward(self, x_ei: torch.Tensor, ca1: bool=False, test: bool=False):

        """
        Forward pass

        Parameters
        ----------
        x_ei: torch.Tensor
            input data
        ca1: bool
            return the data from CA1. Default is False
        test: bool
            Default is False

        Returns
        -------
        torch.Tensor
            reconstructed data
        """

        # forward pass through the entorhinal cortex to CA3
        x_ca3 = self.W_ei_ca3 @ x_ei # 50, 1
        x_ca3 = utils.sparsemoid(x_ca3.reshape(1, -1),
                                 K=self._K_ca3,
                                 beta=self._beta_ca3).reshape(-1, 1)

        # forward pass through CA3 to CA1
        x_ca1 = self.W_ca3_ca1 @ x_ca3 # 50, 1
        x_ca1 = utils.sparsemoid(x_ca1.reshape(1, -1),
                                 K=self._K_lat,
                                 beta=self._beta_ca3,
                                 flag=False).reshape(-1, 1)

        # compute instructive signal
        if self.identity_IS:
            IS = x_ei
        else:
            IS = self.W_ei_ca1 @ x_ei + self.B_ei_ca1
            IS = utils.sparsemoid(IS.reshape(1, -1), K=self._K_lat,
                                  beta=self._beta).reshape(-1, 1)
            if self.random_IS:
                # permute the IS
                IS = IS[torch.randperm(IS.size(0))]

        # weight update
        if self.mode == "train" and not test:
            self.W_ca3_ca1 = nn.Parameter((1 - IS * self._alpha) * \
                self.W_ca3_ca1 + self._alpha * (IS @ x_ca3.T))

        # Forward pass through CA1 to entorhinal cortex output
        x_eo = self.W_ca1_eo @ x_ca1 + self.B_ca1_eo

        # activation function
        x_eo = utils.sparsemoid(x_eo.reshape(1, -1),
                                K=self._K_out,
                                beta=self._beta).reshape(-1, 1)

        self._ca1 = x_ca1
        self._ca3 = x_ca3
        self._eo = x_eo

        self.record(x_ei, IS)
        if ca1:
            return x_eo, x_ca1

        return x_eo

    def pause_lr(self):

        """
        Pause learning rate
        """

        self.mode = "test"

    @property
    def testing_mode(self):
        self.mode = "test"

    def resume_lr(self):

        """
        Resume learning rate
        """

        self.mode = "train"

    @property
    def training_mode(self):
        self.mode = "train"

    def set_alpha(self, alpha: float):
 
        """
        Set the learning rate
        """

        self._alpha = alpha

    def record(self, x_ei, IS):
        self.recordings["x_ei"].append(x_ei.clone())
        self.recordings["ca3"].append(self._ca3.clone())
        self.recordings["ca1"].append(self._ca1.clone())
        self.recordings["eo"].append(self._eo.clone())
        self.recordings["W_ca3_ca1"].append(self.W_ca3_ca1.clone())
        self.recordings["IS"].append(IS.clone())

    def reset(self):

        self._ca1 = None
        self._ca3 = None
        self._eo = None

        # mode
        self.mode = "train"
        self.recordings = {}
        self.recordings["x_ei"] = []
        self.recordings["ca3"] = []
        self.recordings["IS"] = []
        self.recordings["ca1"] = []
        self.recordings["eo"] = []
        self.recordings["W_ca3_ca1"] = []




""" load AE and info """


def load_session(idx: int=None,
                 verbose: bool=True) -> tuple:

    """
    Load the training information and
    the autoencoder model from the saved
    sessions

    Parameters
    ----------
    idx : int
        the index of the session to load.
        Default is None.
    verbose : bool
        print the training information.
        Default is True.

    Returns
    -------
    info : dict
        training information
    model : object
        autoencoder model
    """

    global cache_dir
    global cache_dir_2

    logger.debug(f"loading {idx=}, {cache_dir=} {cache_dir_2=} {os.getcwd()=}")

    # display the saved sessions
    try:
        try:
            ae_sessions = [f for f in os.listdir(cache_dir) if "ae" in f]
        except FileNotFoundError:
            cache_dir = cache_dir_2
            ae_sessions = [f for f in os.listdir(cache_dir) if "ae" in f]
    except FileNotFoundError:
        raise ValueError(f"nor {cache_dir} neither {cache_dir_2} found")

    if len(ae_sessions) == 0:
        raise ValueError("No saved sessions found")

    if idx is None or idx < 0:

        logger("Saved sessions:")
        for i, session in enumerate(ae_sessions):
            print(f"[{i}] {session}")

        # select the session
        idx = int(input("Select session\n>>> "))
    elif verbose:
        logger(f"Pre-selected session: [{idx}]")

    # load the session
    session = ae_sessions[idx]
    with open(f"{cache_dir}/{session}/info.json", "r") as f:
        info = json.load(f)


    if "network_params" in info:
        input_dim = info["network_params"]["dim_ei"]
        encoding_dim = info["network_params"]["dim_ca1"]
        K = info["network_params"]["K_ca1"]
        beta = info["network_params"]["beta_ca1"]
        bias = info["network_params"]["bias"]
    else:
        input_dim = info["dim_ei"]
        encoding_dim = info["dim_ca1"]
        K = info["K_lat"]
        beta = info["beta"]
        try:
            bias = info["bias"]
        except KeyError:
            logger.warning("bias not found in the info file, set to True")
            bias = True

    logger.debug(f"{bias=}")

    # load the model
    model = Autoencoder(input_dim=input_dim,
                        encoding_dim=encoding_dim,
                        activation=None,
                        K=K,
                        beta=beta,
                        use_bias=bias)

    model.load_state_dict(
        torch.load(f"{cache_dir}/{session}/autoencoder.pt"))
        # torch.load(f"{cache_dir}/{session}/autoencoder.pt",
        #            weights_only=True))

    if verbose:
        logger("info:")
        pprint(info)

    return info, model




if __name__ == "__main__":

    # main()

    print("hello")

    # dim_ei = 100
    # dim_ca3 = 200
    # dim_ca1 = 150
    # dim_eo = 100

    # model = MTL(W_ei_ca1=torch.randn(dim_ca1, dim_ei),
    #             W_ca1_eo=torch.randn(dim_eo, dim_ca1),
    #             dim_ca3=dim_ca3, lr=0.9)

    # input_data = torch.randn(dim_ei, 1)  # Batch size of 1 for simplicity

    # with torch.no_grad():
    #     output_data = model(input_data)

    # print(input_data.shape)
    # print(output_data.shape)

    # # Generate some random data
    # N = 100
    # size = 50

    # heads = 3
    # variance = 0.05
    # higher_heads = heads 
    # higher_variance = 0.075

    # samples = utils.stimulus_generator(N, size, heads, variance,
    #                              higher_heads=higher_heads,
    #                              higher_variance=higher_variance,
    #                              plot=False)

    # # make model
    # model = Autoencoder(input_dim=size, encoding_dim=10)

    # # train model
    # epochs = 2e3
    # model = train_autoencoder(samples, model, epochs=int(epochs),
    #                           batch_size=5, learning_rate=1e-3)

    # # reconstruct data
    # reconstruct_data(samples, num=5, model=model)

