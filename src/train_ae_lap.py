import numpy as np
import torch
import argparse
from models import Autoencoder, logger
import utils



""" training settings """

parser = argparse.ArgumentParser(
    description="BTSP model")
parser.add_argument('--num', type=int,
                    help='number of samples',
                    default=500)
parser.add_argument('--epochs', type=int,
                    help='number of epochs',
                    default=1000)
args = parser.parse_args()

NUM_STIMULI = args.num
EPOCHS = args.epochs


""" settings """

network_params = {}

network_params["mec_N_x"] = 50
network_params["mec_N_y"] = 1
network_params["dim_mec"] = network_params["mec_N_x"]*network_params["mec_N_y"]
network_params["mec_sigma"] = 4
network_params["dim_lec"] = 50
network_params["num_cues"] = 2
NUM_CUES = network_params["num_cues"]

network_params["bias"] = False

network_params["dim_ei"] = network_params["dim_mec"] + network_params["dim_lec"]
network_params["dim_ca3"] = 1000
network_params["dim_ca1"] = 1000
network_params["dim_eo"] = network_params["dim_ei"]

network_params["K_lec"] = 5
network_params["K_ei"] = 10
network_params["K_ca3"] = 25
network_params["K_ca1"] = 25
network_params["K_eo"] = network_params["K_ei"]

network_params["beta_ei"] = 150
network_params["beta_ca3"] = 150
network_params["beta_ca1"] = 150
network_params["beta_eo"] = network_params["beta_ei"]

network_params["alpha"] = 0.5

""" make data """

training_samples, _, _ = utils.sparse_stimulus_generator_sensory(
                                num_stimuli=NUM_STIMULI,
                                K = network_params["K_lec"],
                                mec_size=network_params["dim_mec"],
                                lec_size=network_params["dim_lec"],
                                N_x=network_params["mec_N_x"],
                                N_y=network_params["mec_N_y"],
                                pf_sigma=network_params["mec_sigma"],
                                lap_length=network_params["mec_N_x"],
                                num_laps=None,
                                num_cues=None,
                                position_list=None,
                                cue_positions=None,
                                sen_list=None,
                                plot=False)

test_samples, _, _ = utils.sparse_stimulus_generator_sensory(
                                num_stimuli=NUM_STIMULI,
                                K = network_params["K_lec"],
                                mec_size=network_params["dim_mec"],
                                lec_size=network_params["dim_lec"],
                                N_x=network_params["mec_N_x"],
                                N_y=network_params["mec_N_y"],
                                pf_sigma=network_params["mec_sigma"],
                                lap_length=network_params["mec_N_x"],
                                num_laps=None,
                                num_cues=None,
                                position_list=None,
                                cue_positions=None,
                                sen_list=None,
                                plot=False)
logger
logger(f"Training data generated: {training_samples.shape}")


""" autoencoder training """

# autoencoder
autoencoder = Autoencoder(input_dim=network_params["dim_ei"],
                                 encoding_dim=network_params["dim_ca1"],
                                 K=network_params["K_ca1"],
                                 beta=network_params["beta_ca1"],
                                 use_bias=network_params["bias"])
logger(f"%Autoencoder: {autoencoder}")
  
loss_ae, autoencoder = utils.train_autoencoder(
                training_data=training_samples,
                test_data=test_samples,
                model=autoencoder,
                epochs=EPOCHS,
                batch_size=64, learning_rate=1e-3)

loss_ae, autoencoder = utils.train_autoencoder(
                training_data=training_samples,
                test_data=test_samples,
                model=autoencoder,
                epochs=EPOCHS,
                batch_size=64, learning_rate=5e-4)


logger(f"<<< Autoencoder trained [loss={loss_ae:.4f}] >>>")

if str(input("Save model? [y/n]: ")).lower() != "y":

    import sys
    sys.exit(0)


""" save """

# make directory
import os
import datetime
import json


info = {
    "network_params": network_params,
    "num_stimuli": NUM_STIMULI,
    "epochs": EPOCHS,
    "loss_ae": round(loss_ae, 5),
    "date": datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
}

nb = len([name for name in os.listdir("cache") \
    if os.path.isdir(f"cache/{name}") and "ae_" in name])

dir_name = f"cache/ae_{nb}"

os.makedirs(dir_name, exist_ok=True)

# save model
torch.save(autoencoder.state_dict(), f"{dir_name}/autoencoder.pt")

# save info
with open(f"{dir_name}/info.json", "w") as f:
    json.dump(info, f)

logger(f"<<< Model saved in {dir_name} >>>")
