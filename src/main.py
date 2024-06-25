import utils
import torch
from models import Autoencoder, MTL, logger

import matplotlib.pyplot as plt


if __name__ == "__main__":

    """ settings """

    # architecture sizes
    dim_ei = 100
    dim_ca3 = 200
    dim_ca1 = 150
    dim_eo = dim_ei

    # data settings
    nb_samples = 500

    heads = 3
    variance = 0.05
    higher_heads = heads 
    higher_variance = 0.075

    # reconstruction
    num_reconstructions = 5

    # make samples
    training_samples = utils.stimulus_generator(N=nb_samples, size=dim_ei,
                                 heads=heads, variance=variance,
                                 higher_heads=higher_heads,
                                 higher_variance=higher_variance,
                                 plot=False)
    test_samples = utils.stimulus_generator(N=nb_samples, size=dim_ei,
                                 heads=heads, variance=variance,
                                 higher_heads=higher_heads,
                                 higher_variance=higher_variance,
                                 plot=False)
    logger("<<< Data generated >>>")

    """ autoencoder training """

    autoencoder = Autoencoder(input_dim=dim_ei,
                              encoding_dim=dim_ca1)
    logger(f"%Autoencoder: {autoencoder}")

    # train autoencoder
    epochs = 2e1
    loss_ae, autoencoder = utils.train_autoencoder(
                    training_data=training_samples,
                    test_data=test_samples,
                    model=autoencoder,
                    epochs=int(epochs),
                    batch_size=5, learning_rate=1e-3)
    logger(f"<<< Autoencoder trained [loss={loss_ae:.4f}] >>>")

    # reconstruct data
    out_ae = utils.reconstruct_data(data=training_samples,
                                    num=num_reconstructions,
                                    model=autoencoder,
                                    show=False, 
                                    plot=False)

    """ mtl training """

    # get weights from the autoencoder
    W_ei_ca1, W_ca1_eo = autoencoder.get_weights()
    # W_ei_ca1 = torch.randn(dim_ca1, dim_ei)
    # W_ca1_eo = torch.randn(dim_eo, dim_ca1)

    # make model
    model = MTL(W_ei_ca1=W_ei_ca1,
                W_ca1_eo=W_ca1_eo,
                dim_ca3=dim_ca3,
                lr=1.)

    logger(f"%MTL: {model}")

    # train model
    epochs = 1
    for _ in range(epochs):
        loss_mtl, model = utils.testing(data=training_samples,
                                        model=model,
                                        column=True)
        logger(f"<<< MTL trained [{loss_mtl:.3f}] >>>")

    # reconstruct data
    model.pause_lr()
    out_mtl = utils.reconstruct_data(data=training_samples,
                                     num=num_reconstructions,
                                     model=model,
                                     column=True,
                                     plot=False)

    """ plotting """

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 5), sharex=True)

    utils.plot_squashed_data(
        data=training_samples[:num_reconstructions].reshape(num_reconstructions, -1),
                             ax=ax1,
                             title="Original", squash=False)
    utils.plot_squashed_data(data=out_ae, ax=ax2,
                             title="Autoencoder", squash=False)
    utils.plot_squashed_data(data=out_mtl, ax=ax3,
                             title="MTL", squash=False)

    fig.suptitle("Data reconstruction")
    plt.show()

