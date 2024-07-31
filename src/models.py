import torch
from torch import nn

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import utils


logger = utils.setup_logger(__name__)


""" Autoencoder """

class Autoencoder(nn.Module):

    def __init__(self, input_dim: int=10, encoding_dim=10):

        # make docstrings
        """
        Simple autoencoder with a single linear layer as encoder and decoder.

        Parameters
        ----------
        input_dim: int
            the size of the input data
        encoding_dim: int
            the size of the encoded data
        """

        super(Autoencoder, self).__init__()

        self._input_dim = input_dim
        self._encoding_dim = encoding_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            # nn.ReLU(True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            # nn.ReLU(True),
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
        x = self.decoder(z)

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

    def __init__(self, W_ei_ca1: torch.Tensor, W_ca1_eo: torch.Tensor,
                 dim_ca3: int, lr: float):

        # make docstrings
        """
        Multi-target learning model with BTSP learning rule

        Parameters
        ----------
        W_ei_ca1: torch.Tensor
            the weight matrix from entorhinal cortex to CA1
        W_ca1_eo: torch.Tensor
            the weight matrix from CA1 to entorhinal cortex output
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

        #network parameters
        self._lr = lr
        self._lr_orig = lr

        # Initialize weight matrices for each layer
        self.W_ei_ca3 = nn.Parameter(torch.randn(dim_ca3,
                                                 self._dim_ei) / dim_ca3)
        self.W_ei_ca1 = nn.Parameter(W_ei_ca1)
        # self.W_ca3_ca1 = nn.Parameter(torch.randn(self._dim_ca1, dim_ca3))
        self.W_ca3_ca1 = nn.Parameter(torch.zeros(self._dim_ca1, dim_ca3))
        # self.W_ca3_ca1 = nn.Parameter(nn.Linear(self._dim_ca1, dim_ca3,
        #                            bias=False).weight.clone().detach())

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
        x_ca3 = self.W_ei_ca3 @ x_ei

        # Forward pass through CA3 to CA1
        # x_ca1 = torch.matmul(x_ca3, self.W_ca3_ca1)
        x_ca1 = self.W_ca3_ca1 @ x_ca3

        # compute instructive signal
        # IS = torch.matmul(x_ei, self.W_ei_ca1)
        IS = self.W_ei_ca1 @ x_ei

        # ----- # top k values
        # betas = torch.zeros_like(IS)
        # betas[torch.topk(IS.flatten(), Kis).indices] = 1.

        # # betas = betas.reshape(IS.shape)
        # tiled_ca3 = x_ca3.flatten().repeat(self._dim_ca1, 1)
        # self.W_ca3_ca1 = nn.Parameter((1 - betas) * self.W_ca3_ca1 + betas * tiled_ca3)

        # betas = IS | but select the first -k IS
        # betas[torch.topk(IS.flatten(), Kis).indices] = torch.topk(IS.flatten(), Kis).values.flatten()
        # betas = IS
        # self.W_ca3_ca1 = nn.Parameter(x_ca3 @ betas.reshape(1, -1))
        # self.W_ca3_ca1 = nn.Parameter(tiled_ca3 @ betas.T)
        # -----

        # update ca3 -> ca1 connectivity via BTSP
        # W_ca3_ca1_prime  = nn.Parameter(torch.einsum('im,in->imn', x_ca3, IS))
        # self.W_ca3_ca1 = nn.Parameter((1 - self._lr)*self.W_ca3_ca1 + self._lr*W_ca3_ca1_prime)

        W_ca3_ca1_prime  = nn.Parameter(IS @ x_ca3.T)
        self.W_ca3_ca1 += self._lr * W_ca3_ca1_prime

        # ---
        # Forward pass through CA1 to entorhinal cortex output
        # x_eo = torch.matmul(x_ca1, self.W_ca1_eo)
        x_eo = self.W_ca1_eo @ x_ca1

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

