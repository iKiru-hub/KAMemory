import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import sys, os
import warnings
import json
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.path.abspath(__file__).split("src")[0] + "src")
sys.path.append(os.path.abspath(__file__).split("src")[0] + "src/evolution_dir")
import models
import utils
from _utils_ev import *
from evolution_search import *

K_STIMULUS = 3


""" settings """

def run(num_reps: int, num_samples: int, parameters: dict,
        settings: dict):

    K_lat = parameters['K_lat']
    K_ca3 = parameters['K_ca3']
    K_out = parameters['K_out']
    beta_eo = parameters['beta_eo']
    beta_is = parameters['beta_is']
    beta_ca1 = parameters['beta_ca1']
    beta_ca3 =parameters['beta_ca3']
    alpha = parameters['alpha']
    num_swaps_ca1 = parameters["num_swaps_ca1"]
    num_swaps_ca3 = parameters["num_swaps_ca3"]

    # K = settings["K_lat"]
    K = K_STIMULUS # be careful
    dim_ei = settings["dim_ei"]
    W_ei_ca1 = settings["W_ei_ca1"]
    W_ca1_eo = settings["W_ca1_eo"]
    B_ei_ca1 = settings["B_ei_ca1"]
    B_ca1_eo = settings["B_ca1_eo"]

    accuracy = np.zeros((num_reps, num_samples, num_samples))
    sim_ca1_is = np.zeros((num_reps, num_samples, num_samples))
    distances = np.zeros((num_reps, num_samples, num_samples))

    """ run """
    for l in tqdm(range(num_reps), disable=True):

        # data
        stimuli = utils.sparse_stimulus_generator(N=num_samples,
                                                  K=K,
                                                  size=dim_ei,
                                                  plot=False)

        datasets = []
        for k in range(num_samples):
            data = torch.tensor(stimuli[:k+1], dtype=torch.float32)
            dataloader = DataLoader(TensorDataset(data),
                                    batch_size=1,
                                    shuffle=False)
            datasets += [dataloader]

        # run
        for i in tqdm(range(num_samples), disable=True):

            # make model
            model = models.MTLev(W_ei_ca1=W_ei_ca1,
                   W_ca1_eo=W_ca1_eo,
                   K_lat=K_lat,
                   K_out=K_out,
                   K_ca3=K_ca3,
                   dim_ca3=dim_ei,
                   beta_eo=beta_eo,
                   beta_is=beta_is,
                   beta_ca1=beta_ca1,
                   beta_ca3=beta_ca3,
                   alpha=alpha,
                   num_swaps_ca1=num_swaps_ca1,
                   num_swaps_ca3=num_swaps_ca3,
                   B_ei_ca1=B_ei_ca1,
                   B_ca1_eo=B_ca1_eo)

            # train a dataset with pattern index 0.. i
            model.eval()
            with torch.no_grad():

                # one pattern at a time
                for j, batch in enumerate(datasets[i]):
                    # forward
                    _ = model(batch[0].reshape(-1, 1))

                    # to delete?
                    sim_ca1_is[l, i, j] = utils.cosine_similarity_vec(
                        model.recordings['ca1'][-1],
                        model.recordings['IS'][-1])
                    distances[l, i, j] = i-j

            # test a dataset with pattern index 0.. i
            model.pause_lr()
            model.eval()
            with torch.no_grad():
                # one pattern at a time
                for j, batch in enumerate(datasets[i]):
                    x = batch[0].reshape(-1, 1)

                    # forward
                    y = model(x)
                    accuracy[l, i, j] = ((y.T @ x) / (torch.norm(x) * torch.norm(y))).item()

                    sim_ca1_is[l, i, j] = utils.cosine_similarity_vec(
                        model.recordings['ca1'][-1],
                        model.recordings['IS'][-1])
                    distances[l, i, j] = i-j

    return accuracy


def vary_one_p(num_reps: int, num_samples: int,
                parameters: dict, settings: dict,
                grid: np.ndarray, name: str):

    results = []
    for g in grid:
        parameters[name] = g
        results += [run(num_reps, num_samples, parameters, settings)]

    return results


def plot(table: dict):

    nrows = len(table.keys()) - 1
    ncols = len(table[list(table.keys())[0]]["grid"])
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*nrows, ncols*2))
    if nrows == 1:
        axes = [axes]

    for idx, (name, data) in enumerate(table.items()):
        if name == "parameters": continue

        for i, g in enumerate(data["grid"]):
            axes[idx][i].imshow(data["results"][i].mean(axis=0), vmin=0., vmax=1.)
            axes[idx][i].axis('off')
            axes[idx][i].set_title(f"{name}={g:.1f}\n[{data['score'][i]:.2f}]")

    plt.tight_layout()
    plt.show()


def save(data: dict):

    path = os.path.abspath(__file__).split("KAMemory")[0] + \
        "KAMemory/src/data/other"

    now = datetime.now()
    num = len(os.listdir(path))
    name = f"landscape_{num}.json"

    save_data = {}
    for param_name, param_data in data.items():
        if param_name == "parameters":
            save_data["parameters"] = param_data
            continue

        save_data[param_name] = {
            "grid": param_data["grid"].tolist(),
            "score": param_data["score"],
        }

    save_data["K"] = K_STIMULUS

    with open(os.path.join(path, name), "w") as f:
        json.dump(save_data, f, indent=2)

    print()
    logger(f"saved to {path}/{name}")


# ==================================================================
# ==================================================================


def run_three_params(num_reps: int, num_samples: int, parameters: dict,
                     settings: dict, param1_name: str, param1_grid: np.ndarray,
                     param2_name: str, param2_grid: np.ndarray,
                     param3_name: str, param3_grid: np.ndarray):
    """
    Run a 3D parameter sweep over three parameters.
    Returns a 3D array of scores (shape: len(param1_grid) x len(param2_grid) x len(param3_grid)).
    """
    scores = np.zeros((len(param1_grid), len(param2_grid), len(param3_grid)))

    for i, g1 in enumerate(tqdm(param1_grid, desc=param1_name, leave=False)):
        for j, g2 in enumerate(tqdm(param2_grid, desc=param2_name, leave=False)):
            for k, g3 in enumerate(tqdm(param3_grid, desc=param3_name, leave=False)):
                # Set parameters
                parameters[param1_name] = g1
                parameters[param2_name] = g2
                parameters[param3_name] = g3
                # Run model
                _data = run(num_reps, num_samples, parameters, settings)
                # Compute score (same as in main)
                scores[i, j, k] = np.mean(exp_eval(data=_data.mean(axis=0), sigma=settings["sigma"]))

    return scores


def plot_3d(scores: np.ndarray, param1_name: str, param1_grid: np.ndarray,
            param2_name: str, param2_grid: np.ndarray,
            param3_name: str, param3_grid: np.ndarray):
    """
    Plot 3D results as a series of 2D heatmaps (one for each value of the third parameter).
    """
    nrows = len(param3_grid)
    ncols = 1  # We'll make a single column of heatmaps, one per param3 value
    fig, axes = plt.subplots(nrows, ncols, figsize=(6, 4*nrows))
    if nrows == 1:
        axes = [axes]

    for k, g3 in enumerate(param3_grid):
        # Extract 2D slice for this value of param3
        slice_scores = scores[:, :, k]
        im = axes[k].imshow(slice_scores, aspect='auto', origin='lower',
                           extent=[param1_grid[0], param1_grid[-1], param2_grid[0], param2_grid[-1]],
                           vmin=0, vmax=1)
        axes[k].set_xlabel(param1_name)
        axes[k].set_ylabel(param2_name)
        axes[k].set_title(f"{param3_name} = {g3:.2f}")
        plt.colorbar(im, ax=axes[k])

    plt.tight_layout()
    plt.show()


def save_3d(scores: np.ndarray, param1_name: str, param1_grid: np.ndarray,
            param2_name: str, param2_grid: np.ndarray,
            param3_name: str, param3_grid: np.ndarray,
            settings: dict, parameters: dict):
    """
    Save 3D results to a JSON file.
    """
    path = os.path.abspath(__file__).split("KAMemory")[0] + \
        "KAMemory/src/data/other"

    now = datetime.now()
    num = len(os.listdir(path))
    name = f"landscape_3d_{num}.json"

    save_data = {
        "parameters": parameters,
        param1_name: {
            "grid": param1_grid.tolist(),
            "label": param1_name
        },
        param2_name: {
            "grid": param2_grid.tolist(),
            "label": param2_name
        },
        param3_name: {
            "grid": param3_grid.tolist(),
            "label": param3_name
        },
        "scores": scores.tolist(),
        "K": K_STIMULUS,
        "sigma": settings["sigma"]
    }

    with open(os.path.join(path, name), "w") as f:
        json.dump(save_data, f, indent=2)

    print()
    logger(f"saved to {path}/{name}")


def main(num_samples: int, num_reps: int, grids: list, load_index: int=0):

    # -- settings
    settings = load_autoencoder(index=load_index)
    settings["num_samples"] = num_samples
    settings["sigma"] = SIGMA
    settings["genome_configs"] = GENOME_CONFIGS

    results = {}
    for grid in tqdm(grids, desc="=== params"):
        name = grid["name"]
        grid_vals = grid["grid"]

        pbar = tqdm(grid_vals, desc=f"{name}", leave=True,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} " + \
                        "[{elapsed}<{remaining}, {rate_fmt}]")

        run_results = []
        scores = []
        for g in pbar:

            # baseline parameters
            parameters = {"K_lat": 15,
                          "K_ca3": 6.,
                          "K_out": settings["K_out"],
                          "beta_ca3": 28.,
                          "beta_ca1": 105.,
                          "beta_is": 210.,
                          "beta_eo": 125.,
                          "alpha": 0.1,
                          "num_swaps_ca1": 1.0,
                          "num_swaps_ca3": 1.0}

            parameters[name] = g
            _data = run(num_reps, num_samples, parameters, settings)
            scores += [np.mean(exp_eval(data=_data.mean(axis=0),
                                      sigma=SIGMA))]
            run_results.append(_data)

        results[name] = {"results": run_results, "grid": grid_vals,
                         "score": scores}

    results["parameters"] = parameters
    save(results)
    plot(results)


def main_3d(num_samples: int, num_reps: int, load_index: int=0):
    """
    Main function for 3D parameter sweep.
    """
    # -- settings
    settings = load_autoencoder(index=load_index)
    settings["num_samples"] = num_samples
    settings["sigma"] = SIGMA
    settings["genome_configs"] = GENOME_CONFIGS

    # Define three parameters to sweep (example: alpha, beta_is, beta_ca1)
    num = 8
    param1_name = "alpha"
    param1_grid = np.linspace(0.1, 0.2, 2)

    param2_name = "beta_is"
    param2_grid = np.linspace(1., 300, num)

    param3_name = "beta_ca3"
    param3_grid = np.linspace(1., 300, num)

    # Baseline parameters
    parameters = {"K_lat": 15,
                  "K_ca3": 6.,
                  "K_out": settings["K_out"],
                  "beta_ca3": 28.,
                  "beta_ca1": 105.,
                  "beta_is": 210.,
                  "beta_eo": 125.,
                  "alpha": 0.1,
                  "num_swaps_ca1": 1.0,
                  "num_swaps_ca3": 1.0}

    # Run 3D sweep
    print("Running 3D parameter sweep...")
    scores = run_three_params(num_reps, num_samples, parameters, settings,
                              param1_name, param1_grid,
                              param2_name, param2_grid,
                              param3_name, param3_grid)

    # Plot results
    print("Plotting results...")
    plot_3d(scores, param1_name, param1_grid, param2_name, param2_grid, param3_name, param3_grid)

    # Save results
    print("Saving results...")
    save_3d(scores, param1_name, param1_grid, param2_name, param2_grid, param3_name, param3_grid, settings, parameters)


if __name__ == "__main__":

    GRID_SIZE = 12

    grids = [

        {"name": "alpha",
         "grid": np.linspace(0.01, 0.4, GRID_SIZE )},

        {"name": "beta_is",
         "grid": np.linspace(1., 300, GRID_SIZE )},

        {"name": "beta_ca1",
         "grid": np.linspace(1., 300, GRID_SIZE )},

        {"name": "beta_ca3",
         "grid": np.linspace(1., 300, GRID_SIZE )},

        # {"name": "K_ca3",
        #  "grid": np.linspace(1, 30, GRID_SIZE )},

        # {"name": "K_lat",
        #  "grid": np.linspace(1, 30, GRID_SIZE )},

        # {"name": "num_swaps_ca1",
        #  "grid": np.linspace(1., 10., GRID_SIZE )},

        # {"name": "num_swaps_ca3",
        #  "grid": np.linspace(1., 10., GRID_SIZE )},

    ]

    # main(num_samples=48, num_reps=1, grids=grids)
    main_3d(num_samples=32, num_reps=1)

    print("\n[done]")
