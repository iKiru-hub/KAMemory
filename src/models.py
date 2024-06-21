import torch
from torch import nn
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import utils



class Autoencoder(nn.Module):

    def __init__(self, input_size: int=10, encoding_dim=10):

        # make docstrings
        """
        Simple autoencoder with a single linear layer as encoder and decoder.

        Parameters
        ----------
        input_size: int
            the size of the input data
        encoding_dim: int
            the size of the encoded data
        """

        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, encoding_dim),
            # nn.ReLU(True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_size),
            # nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor):

        """
        Forward pass

        Parameters
        ----------
        x: torch.Tensor
            input data

        Returns
        -------
        torch.Tensor
            reconstructed data
        """

        x = self.encoder(x)
        x = self.decoder(x)

        return x


def train_autoencoder(data: np.ndarray, model: object,
                      epochs: int=20, batch_size: int=64,
                      learning_rate: float=1e-3):

    """
    Train the autoencoder model

    Parameters
    ----------
    data: np.ndarray
        input data
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
    data_tensor = torch.tensor(data, dtype=torch.float32)

    # Create a dataset and data loader
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Set the model to training mode
    model.train()

    # Training loop
    epoch = 0
    epoch_log = 100
    for epoch in (pbar := tqdm(range(epochs), desc = f"{epoch}")):
        total_loss = 0
        for batch in dataloader:
            inputs = batch[0]

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch+1) % epoch_log == 0:
            pbar.set_description(f"Epoch [{epoch+1}], Loss: {total_loss / len(dataloader):.4f}")

    return model


def reconstruct_data(data: np.ndarray, model: object, num: int=5):

    """
    Reconstruct data using the autoencoder model

    Parameters
    ----------
    data: np.ndarray
        input data
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

    # Reconstruct data
    reconstructed_data = []
    with torch.no_grad():

        for batch in tqdm(dataloader):

            inputs = batch[0]

            # Forward pass
            outputs = model(inputs)
            reconstructed_data.append(outputs.numpy().flatten())

    # Convert list to numpy array
    reconstructed_data = np.array(reconstructed_data)

    # difference between original and reconstructed data
    diff_data = (data[:num] - reconstructed_data)

    # plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(data_tensor, aspect="auto", vmin=0, vmax=1, cmap="gray_r")
    ax1.set_title("Original data")
    ax1.set_axis_off()

    ax2.imshow(reconstructed_data, aspect="auto", vmin=0, vmax=1, cmap="gray_r")
    ax2.set_title("Reconstructed data")
    ax2.set_axis_off()

    ax3.imshow(diff_data, aspect="auto", cmap="seismic", vmin=-1, vmax=1)
    ax3.set_title("Difference")
    ax3.set_axis_off()

    plt.show()

    return reconstructed_data



if __name__ == "__main__":

    # Generate some random data
    N = 200
    input_size = 50

    heads = 3
    variance = 0.01
    higher_heads = heads
    higher_variance = 0.05

    samples = utils.stimulus_generator(
        N, input_size, heads, variance,
        higher_heads=higher_heads,
        higher_variance=higher_variance,
        plot=False)

    """
    TODO:
    - train-test split
    """

    # make model
    model = Autoencoder(input_size=input_size,
                        encoding_dim=input_size)

    print(model)

    # train model
    epochs = 2_000
    model = train_autoencoder(samples,
                              model,
                              epochs=int(epochs),
                              batch_size=5,
                              learning_rate=1e-3)


    # reconstruct data -- new data
    samples = utils.stimulus_generator(
        1000, input_size,
        heads, variance,
        higher_heads=higher_heads,
        higher_variance=higher_variance,
        plot=False)

    x_rec = reconstruct_data(samples, num=10, model=model)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(samples.sum(axis=0).reshape(1, -1),
               aspect="auto", cmap="gray_r")
    ax1.set_title("Original higher distribution")

    ax2.imshow(x_rec.sum(axis=0).reshape(1, -1),
               aspect="auto", cmap="gray_r")
    ax2.set_title("Reconstructed higher distribution")

    plt.show()


