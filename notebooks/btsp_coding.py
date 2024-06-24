# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% id="1WuYtba2S81N"
import torch
import torch.nn as nn


# %% id="s9WTh_XtTPZF"
class MTL(nn.Module):
    def __init__(self, n_entorhinal_in, n_ca3, n_ca1, n_entorhinal_out):
        super(MTL, self).__init__()

        #network parameters
        self.beta_btsp = 0.9

        # Initialize weight matrices for each layer
        self.W_ei_ca3 = nn.Parameter(torch.randn(n_entorhinal_in, n_ca3))
        self.W_ei_ca1 = nn.Parameter(torch.randn(n_entorhinal_in, n_ca1))
        self.W_ca3_ca1 = nn.Parameter(torch.randn(n_ca3, n_ca1))
        self.W_ca1_eo = nn.Parameter(torch.randn(n_ca1, n_entorhinal_out))

    def forward(self, x_ei):
        # Forward pass through the entorhinal cortex to CA3
        x_ca3 = torch.matmul(x_ei, self.W_ei_ca3)
        #x = torch.relu(x)  # Activation function (ReLU)

        # Forward pass through CA3 to CA1
        x_ca1 = torch.matmul(x_ca3, self.W_ca3_ca1)
        #x = torch.relu(x)  # Activation function (ReLU)

        #compute instructive signal
        IS = torch.matmul(x_ei, self.W_ei_ca1)

        #update ca3 -> ca1 connectivity via BTSP
        W_ca3_ca1_prime  = nn.Parameter(torch.einsum('im,in->imn', x_ca3, IS))
        self.W_ca3_ca1 = nn.Parameter((1 - self.beta_btsp)*self.W_ca3_ca1 + self.beta_btsp*W_ca3_ca1_prime)

        # Forward pass through CA1 to entorhinal cortex output
        x_eo = torch.matmul(x_ca1, self.W_ca1_eo)
        #x = torch.relu(x)  # Activation function (ReLU)

        return x_eo

# %% id="FFiR4dwOTb0H"
# Example usage
n_entorhinal_in = 100
n_ca3 = 200
n_ca1 = 150
n_entorhinal_out = 100

network = MTL(n_entorhinal_in, n_ca3, n_ca1, n_entorhinal_out)

# %% colab={"base_uri": "https://localhost:8080/"} id="JtykRbucTcvD" outputId="d5412244-ecdf-4a14-b1e5-24cdffb8a622"
#input_data = torch.randn(1, n_entorhinal_in)  # Batch size of 1 for simplicity
output_data = network(input_data)
print(output_data)

# %% id="AsAUSmyNdQYD"
