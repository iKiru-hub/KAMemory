import numpy as np
import torch
from models import Autoencoder, logger
import os, json
from pprint import pprint
import argparse

import utils
import training


# ==============================================================================


def main(save: bool, configs: dict, config_key: int):

    """
    train and test an autoencoder from scratch

    Parameters
    ----------
    save : bool
        flag for saving the results
    configs : dict
        loaded settings
    config_key : bool
        type of loaded settings
    """

    # -- make stimuli
    if config_key == 0:
        training_samples = utils.sparse_stimulus_generator(N=configs['ae_configs']['num_stimuli'],
                                                           K=configs['hyperparameters']['K'],
                                                           size=configs['hyperparameters']['dim_ei'],
                                                           plot=False)
        test_samples = utils.sparse_stimulus_generator(N=configs['ae_configs']['num_stimuli'],
                                                       K=configs['hyperparameters']['K'],
                                                       size=configs['hyperparameters']['dim_ei'],
                                                       plot=False)
    elif config_key == 1:
        training_samples, _, _ = utils.sparse_stimulus_generator_sensory(
                                        num_stimuli=configs['ae_configs']['num_stimuli'],
                                        K = configs['hyperparameters']["K_lec"],
                                        mec_size=configs['hyperparameters']["dim_mec"],
                                        lec_size=configs['hyperparameters']["dim_lec"],
                                        N_x=configs['hyperparameters']["mec_N_x"],
                                        N_y=configs['hyperparameters']["mec_N_y"],
                                        pf_sigma=configs['hyperparameters']["mec_sigma"],
                                        lap_length=configs['hyperparameters']["mec_N_x"],
                                        num_laps=None,
                                        num_cues=None,
                                        position_list=None,
                                        cue_positions=None,
                                        sen_list=None,
                                        plot=False)

        test_samples, _, _ = utils.sparse_stimulus_generator_sensory(
                                        num_stimuli=configs['ae_configs']['num_stimuli'],
                                        K = configs['hyperparameters']["K_lec"],
                                        mec_size=configs['hyperparameters']["dim_mec"],
                                        lec_size=configs['hyperparameters']["dim_lec"],
                                        N_x=configs['hyperparameters']["mec_N_x"],
                                        N_y=configs['hyperparameters']["mec_N_y"],
                                        pf_sigma=configs['hyperparameters']["mec_sigma"],
                                        lap_length=configs['hyperparameters']["mec_N_x"],
                                        num_laps=None,
                                        num_cues=None,
                                        position_list=None,
                                        cue_positions=None,
                                        sen_list=None,
                                        plot=False)

    logger(f"Training data generated: {training_samples.shape}")


    # -- declare autoencoder
    beta = configs['hyperparameters']['beta'] if CONFIG_KEY == 0 else configs['hyperparameters']['beta_ca1']
    K_lat = configs['hyperparameters']['K_lat'] if CONFIG_KEY == 0 else configs['hyperparameters']['dim_ca1']
    autoencoder = Autoencoder(input_dim=configs['hyperparameters']['dim_ei'],
                              encoding_dim=configs['hyperparameters']['dim_ca1'],
                              K=K_lat,
                              beta=beta)
    logger(f"%Autoencoder: {autoencoder}")

    # -- train autoencoder
    logger("training..")
    loss_ae, autoencoder = training.train_autoencoder(
                                            training_data=training_samples,
                                            test_data=test_samples,
                                            model=autoencoder,
                                            epochs=int(configs['ae_configs']['epochs']),
                                            batch_size=configs['ae_configs']['batch_size'],
                                            learning_rate=configs['ae_configs']['learning_rate'])

    # --
    print()
    logger(f"Autoencoder trained [loss={loss_ae:.4f}]")

    if save: save_model()

    logger("[done]")


# ==============================================================================


if __name__ == '__main__':

    # -- fetch user arguments
    parser = argparse.ArgumentParser(description="training Autoencoder")
    parser.add_argument('--key', type=int,
                        help='0: normal, 1: lap data',
                        default=0)
    parser.add_argument('--save', action='store_true',
                        help='verbose', default=False)
    args = parser.parse_args()

    # --
    logger(f"-- @{__file__} --")

    logger("training an Autoencoder from scratch")
    CONFIG_KEY = args.key

    if CONFIG_KEY == 0:
        with open(utils.CONFIGS_PATH + "base_configs.json", "r") as f:
            configs = json.load(f)
        logger("%base configs")

    elif CONFIG_KEY == 1:
        with open(utils.CONFIGS_PATH + "lap_configs.json", "r") as f:
            configs = json.load(f)
        logger("%lap configs")

    logger("configs:")
    pprint(configs)
    logger(f"SAVE={args.save}")

    # --
    main(save=args.save, configs=configs, config_key=CONFIG_KEY)

