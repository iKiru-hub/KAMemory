import torch
import numpy as np
import argparse

import matplotlib.pyplot as plt

try:
    import utils
    from models import Autoencoder, MTL, logger, load_session
except ModuleNotFoundError:
    try:
        import src.utils as utils
        from src.models import Autoencoder, MTL, logger, load_session
    except ModuleNotFoundError:
        raise ValueError("`utils` module not found")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="BTSP model")
    parser.add_argument('--num', type=int,
                        help='number of samples',
                        default=1)
    parser.add_argument('--load', action='store_true',
                        help='load session')
    parser.add_argument('--idx', type=int,
                        help='number of samples',
                        default=0)
    args = parser.parse_args()

    """ settings """

    if args.load:

        info, autoencoder = load_session(idx=args.idx)

        dim_ei = info["dim_ei"]
        dim_ca3 = info["dim_ca3"]
        dim_ca1 = info["dim_ca1"]
        dim_eo = info["dim_eo"]

        num_samples = info["num_samples"]

        K_lat = info["K_lat"]
        beta = info["beta"]
        K = info["K"]

        logger("<<< Loaded session >>>")

    # make new settings
    if not args.load:

        dim_ei = 50
        dim_ca3 = 50 
        dim_ca1 = 50
        dim_eo = dim_ei

        # data settings
        num_samples = 300

        # model hyper-parameters
        K = 5
        K_lat = 15
        beta = 60

        # autoencoder
        autoencoder = Autoencoder(input_dim=dim_ei,
                                  encoding_dim=dim_ca1,
                                  activation=None,
                                  K=K_lat,
                                  beta=beta)
        logger(f"%Autoencoder: {autoencoder}")

    """ make data """

    training_samples = utils.sparse_stimulus_generator(N=num_samples,
                                                       K=K,
                                                       size=dim_ei,
                                                       plot=False)
    test_samples = utils.sparse_stimulus_generator(N=num_samples,
                                                   K=K,
                                                   size=dim_ei,
                                                   plot=False)

    # dataset for btsp
    num_btsp_samples = args.num
    training_sample_btsp = training_samples[np.random.choice(
                            range(training_samples.shape[0]),
                            num_btsp_samples, replace=False)]


    logger("<<< Data generated >>>")

    """ autoencoder training """

    # train autoencoder
    if not args.load:
        epochs = 400
        loss_ae, autoencoder = utils.train_autoencoder(
                        training_data=training_samples,
                        test_data=test_samples,
                        model=autoencoder,
                        epochs=int(epochs),
                        batch_size=10, learning_rate=1e-3)
        logger(f"<<< Autoencoder trained [loss={loss_ae:.4f}] >>>")

    # reconstruct data
    out_ae, latent_ae = utils.reconstruct_data(data=training_sample_btsp,
                                    num=num_btsp_samples,
                                    model=autoencoder,
                                    show=False, 
                                    plot=False)

    """ mtl training """

    # get weights from the autoencoder
    # W_ei_ca1, W_ca1_eo = autoencoder.get_weights()
    W_ei_ca1, W_ca1_eo, B_ei_ca1, B_ca1_eo = autoencoder.get_weights(bias=True)

    logger.debug(f"{W_ei_ca1.shape=}\n{type(W_ei_ca1)}")

    # make model
    model = MTL(W_ei_ca1=W_ei_ca1,
                W_ca1_eo=W_ca1_eo,
                B_ei_ca1=B_ei_ca1,
                B_ca1_eo=B_ca1_eo,
                dim_ca3=dim_ca3,
                K_lat=K_lat,
                K_out=K,
                beta=beta)

    logger(f"%MTL: {model}")

    # train model | testing = training without backprop
    epochs = 1
    for _ in range(epochs):
        _, model = utils.testing(data=training_sample_btsp,
                                 model=model,
                                 column=True)

        loss_mtl, _ = utils.testing(data=training_sample_btsp,
                                    model=model,
                                    column=True)
        logger(f"<<< MTL trained [{loss_mtl:.3f}] >>>")

    # reconstruct data
    model.pause_lr()
    out_mtl, latent_mtl = utils.reconstruct_data(
                     data=training_sample_btsp,
                     num=num_btsp_samples,
                     model=model,
                     column=True,
                     plot=False)

    """ plotting """

    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 5), sharex=True)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 5), sharex=True)
    is_squash = False

    utils.plot_squashed_data(
        data=training_sample_btsp,
                             ax=ax1,
                             title="Patterns", squash=is_squash)
    utils.plot_squashed_data(data=latent_ae, ax=ax2,
                             title="Autoencoder",
                             squash=is_squash)
    utils.plot_squashed_data(data=latent_mtl, ax=ax3,
                             title="MTL", squash=is_squash)

    fig.suptitle(f"Latent layers - $K_l=${K_lat} $\\beta=${autoencoder._beta}")

    #
    fig2, (ax12, ax22, ax32) = plt.subplots(3, 1, figsize=(15, 5), sharex=True)
    is_squash = False

    utils.plot_squashed_data(
        data=training_sample_btsp,
                             ax=ax12,
                             title="Patterns", squash=is_squash)
    utils.plot_squashed_data(data=out_ae, ax=ax22,
                             title="Autoencoder",
                             squash=is_squash)
    utils.plot_squashed_data(data=out_mtl, ax=ax32,
                             title="MTL", squash=is_squash)

    # print("AE: ", np.around(out_ae, 2))
    # print("MTL: ", np.around(out_mtl, 2))

    fig2.suptitle(f"Data reconstruction of {num_btsp_samples} patterns - $K=${K} $\\beta=${autoencoder._beta}",
                  fontsize=15)

    #
    fig3, (ax13) = plt.subplots(1, 1, figsize=(15, 5), sharex=True)
    # colorbar
    # cbar = plt.colorbar(
    #     ax13.imshow(model.W_ca3_ca1.detach().numpy(),
    #                 cmap="Greys",
    #                 aspect="auto"))

    cbar = plt.colorbar(
        ax13.imshow(training_sample_btsp - out_mtl,
                    cmap="seismic",
                    aspect="auto"))
    ax13.set_yticks(range(num_btsp_samples))

    cbar.set_label("Error")
    ax13.set_title("pattern - mtl")
    plt.show()


    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 5), sharex=True)

    # ax1.imshow(latent_ae[:5], cmap='gray_r', aspect='auto')
    # ax1.set_ylabel("Autoencoder")
    # print(f"\n>>> latent_ae (last ECin input): {np.around(latent_ae[-1], 2)}")

    # ax2.imshow(latent_mtl[:5], cmap='gray_r', aspect='auto')
    # ax2.set_ylabel("MTL")
    # print(f"\n>>> latent_mtl (last ECin input): {np.around(latent_mtl[-1], 2)}")

    # ax3.imshow(latent_mtl_rnd[:5], cmap='gray_r', aspect='auto')
    # ax3.set_ylabel("MTL (random)")
    # print(f"\n>>> latent_mtl_rnd (last ECin input): {np.around(latent_mtl_rnd[-1], 2)}")

    # fig.suptitle("Latent space [CA1]")
    # plt.show()
