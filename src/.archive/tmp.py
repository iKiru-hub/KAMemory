import json






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

configs = {
    "hyperparameters": network_params,
    "ae_configs": {"num_stimuli": 20,
                    "epochs": 20,
                    "batch_size": 1,
                    "learning_rate": 0.001}
}


with open("configs/lap_configs.json", "w") as f:
    json.dump(configs, f)




configs2= {
    "hyperparameters": {
                      "dim_ei": 50,
                      "dim_ca3": 50,
                      "dim_ca1": 50,
                      "dim_eo": 50,
                      "K": 5,
                      "K_lat": 18,
                      "beta": 54,
        },
    "ae_configs": {
              "num_stimuli": 20,
              "epochs": 20,
              "batch_size": 1,
              "learning_rate": 0.001
    }
}

with open("configs/base_configs.json", "w") as f:
    json.dump(configs2, f)

print("done")
