import torch
from torch import nn

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, json
from pprint import pprint

import utils


logger = utils.setup_logger(__name__)

cache_dir = "cache"
cache_dir_2 = "src/cache"


""" Autoencoder """

class Autoencoder(nn.Module):

    def __init__(self, input_dim: int=10, encoding_dim=10,
                 activation: str=None, K: int=10, beta: float=20.):

        """
        Simple autoencoder with a single linear layer as encoder and decoder.

        Parameters
        ----------
        input_dim: int
            the size of the input data
        encoding_dim: int
            the size of the encoded data
        activation: str
            the activation function to use, choices are
            [None, sparsemax, sigmoid].
            Default is None
        """

        super(Autoencoder, self).__init__()

        self._input_dim = input_dim
        self._encoding_dim = encoding_dim
        self._K = K
        self._beta = beta

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim, bias=True),
            # nn.ReLU(True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim, bias=True),
            # nn.ReLU(True),
        )

        # if activation == "sparsemax":
        #     self.encoder.add_module("sparsemax",
        #                             utils.Sparsemax())
        #     self.decoder.add_module("sparsemax",
        #                             utils.Sparsemax())
        # elif activation == "sigmoid":
        #     self.encoder.add_module("sigmoid",
        #                             nn.Sigmoid())
        #     self.decoder.add_module("sigmoid",
        #                             nn.Sigmoid())
        # elif activation == "soft":
        #     self.encoder.add_module("soft",
        #                             utils.SoftSigmoid())
        #     self.decoder.add_module("soft",
        #                             utils.SoftSigmoid())

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

        # print(z)

        # --- activation function | 1st 2nd 3rd... [Kth Kth+1] ... last
        # z_sorted = torch.sort(z, descending=True, dim=1).values

        # alpha = z_sorted[:, self._K:self._K+2]
        # alpha = alpha.mean(axis=1).reshape(-1, 1)

        # apply
        # z = self._beta * (z - alpha)
        # z = self._beta * z
        z = utils.sparsemoid(z=z, K=self._K,
                             beta=self._beta)

        # z = torch.sigmoid(z)
        # ---

        x = self.decoder(z)
        # x = utils.sparsemoid(x, K=self._K,
        #                      beta=self._beta)
        x = torch.sigmoid(10*(x-0.1))

        if ca1:
            return x, z

        return x

    def get_weights(self):

        """
        Get the weights of the autoencoder model

        Returns
        -------
        tuple
            the weights of the encoder and decoder
        """

        ei_ca1 = self.encoder[0].weight.data.reshape(self._encoding_dim, self._input_dim)
        ca1_eo = self.decoder[0].weight.data.reshape(self._input_dim, self._encoding_dim)

        return ei_ca1, ca1_eo


""" Main model """


Kis = 50


class MTL(nn.Module):

    def __init__(self, W_ei_ca1: torch.Tensor,
                 W_ca1_eo: torch.Tensor,
                 K_lat: int,
                 K_out: int,
                 beta: float,
                 dim_ca3: int,
                 lr: float, activation: str=None):

        # make docstrings
        """
        Multi-target learning model with BTSP learning rule

        Parameters
        ----------
        W_ei_ca1: torch.Tensor
            the weight matrix from entorhinal cortex to CA1
        W_ca1_eo: torch.Tensor
            the weight matrix from CA1 to entorhinal cortex output
        K_lat: int
            the number of top values to select
        K_out: int
            the number of top values to select for the output
        beta: float
            the beta value for the sparsemoid function
        dim_ca3: int
            the size of the CA3 layer
        lr: float
            the learning rate
        """

        super(MTL, self).__init__()

        # infer dimensions of EC input and output and CA1
        self._dim_ei = W_ei_ca1.shape[1]
        self._dim_eo = W_ca1_eo.shape[0]
        self._dim_ca1 = W_ca1_eo.shape[1]

        # network parameters
        self._lr = lr
        self._lr_orig = lr
        self._K_lat = K_lat
        self._K_out = K_out
        self._beta = beta

        # activation function
        if activation == "sparsemax":
            self.activation = utils.Sparsemax()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "soft":
            self.activation = utils.SoftSigmoid()
        else:
            self.activation = utils.Identity()

        # Initialize weight matrices for each layer
        self.W_ei_ca3 = nn.Parameter(torch.randn(dim_ca3,
                                                 self._dim_ei) / dim_ca3)
        self.W_ei_ca1 = nn.Parameter(W_ei_ca1)
        self.W_ca3_ca1 = nn.Parameter(torch.zeros(self._dim_ca1, dim_ca3))

        self.W_ca1_eo = nn.Parameter(W_ca1_eo)

        self._ca1 = None
        self._ca3 = None

    def __repr__(self):

        return f"MTL(dim_ei={self._dim_ei}, dim_ca1={self._dim_ca1}, dim_ca3={self.W_ei_ca3.shape[0]}, dim_eo={self._dim_eo})"

    def forward(self, x_ei: torch.Tensor, ca1: bool=False):

        """
        Forward pass

        Parameters
        ----------
        x_ei: torch.Tensor
            input data
        ca1: bool
            return the data from CA1. Default is False

        Returns
        -------
        torch.Tensor
            reconstructed data
        """

        # Forward pass through the entorhinal cortex to CA3
        # x_ca3 = torch.matmul(x_ei, self.W_ei_ca3)
        x_ca3 = self.W_ei_ca3 @ x_ei # 50, 1
        x_ca3 = utils.sparsemoid(x_ca3.reshape(1, -1),
                                 K=2,
                                 beta=self._beta).reshape(-1, 1)

        # activation function
        # x_ca3 = self.activation(x_ca3)

        # --- implement 

        # Forward pass through CA3 to CA1
        # x_ca1 = torch.matmul(x_ca3, self.W_ca3_ca1)
        x_ca1 = self.W_ca3_ca1 @ x_ca3 # 50, 1

        # print("ei ", np.around(x_ei.T, 2))
        # print("[ca1] ", x_ca1.shape)
        # print("w31 ", np.around(self.W_ca3_ca1, 2))

        # -- x=(5, 50)
        x_ca1 = utils.sparsemoid(x_ca1.reshape(1, -1),
                                 K=self._K_lat,
                                 beta=self._beta,
                                 flag=False).reshape(-1, 1)

        # compute instructive signal
        IS = self.W_ei_ca1 @ x_ei

        # activation function
        IS = utils.sparsemoid(IS.reshape(1, -1), K=self._K_lat,
                              beta=self._beta).reshape(-1, 1)

        # ----- # top k values
        alpha = 0.01
        if self._lr > 0:
            self.W_ca3_ca1 = nn.Parameter((1 - IS * alpha) * \
                self.W_ca3_ca1 + alpha * (IS @ x_ca3.T))

        # Forward pass through CA1 to entorhinal cortex output
        x_eo = self.W_ca1_eo @ x_ca1

        # activation function
        x_eo = utils.sparsemoid(x_eo.reshape(1, -1),
                                K=self._K_out,
                                beta=self._beta).reshape(-1, 1)

        self._ca1 = x_ca1
        self._ca3 = x_ca3

        if ca1:
            return x_eo, x_ca1

        return x_eo

    def pause_lr(self):

        """
        Pause learning rate
        """

        self._lr = 0.

    def resume_lr(self):

        """
        Resume learning rate
        """

        self._lr = self._lr_orig


""" load AE and info """


def load_session(idx: int=None) -> tuple:

    """
    Load the training information and
    the autoencoder model from the saved
    sessions

    Parameters
    ----------
    idx : int
        the index of the session to load.
        Default is None

    Returns
    -------
    info : dict
        training information
    model : object
        autoencoder model
    """

    global cache_dir
    global cache_dir_2

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

    logger("Saved sessions:")
    for i, session in enumerate(ae_sessions):
        print(f"[{i}] {session}")

    if idx is None or idx < 0:
        # select the session
        idx = int(input("Select session\n>>> "))
    else:
        logger(f"Pre-selected session: [{idx}]")

    # load the session
    session = ae_sessions[idx]
    with open(f"{cache_dir}/{session}/info.json", "r") as f:
        info = json.load(f)

    # load the model
    model = Autoencoder(input_dim=info["dim_ei"],
                        encoding_dim=info["dim_ca1"],
                        activation=None,
                        K=info["K_lat"],
                        beta=info["beta"])
    model.load_state_dict(torch.load(f"{cache_dir}/{session}/autoencoder.pt"))

    logger("info:")
    pprint(info)

    return info, model




if __name__ == "__main__":

    main()

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

