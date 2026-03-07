import numpy as np
import matplotlib.pyplot as plt
import sys, os
import argparse
from tqdm import tqdm
import time

sys.path.append(os.path.abspath(__file__).split("src")[0] + "src")
import models
import utils
from logger import logger
from _utils_ev import *

sys.path.append(os.getcwd().split("doki")[0] + \
    "doki/side_lab/evolution_cpp/core/cpp1/build")
import evolution as ev


"""
TODO:

[x] add save option


GENOME:
index=0 : K_lat
index=1 : K_ca3
index=2 : K_out
index=3 : beta
index=4 : alpha

"""

COLORS = ("red", "black", "blue", "green", "orange", "brown")

# --------------------------------------------------
# --------------------------------------------------

def fit_population(population: list, datasets: list, settings: dict):
    return [[ np.exp(-1. * np.clip(evaluate_genome(ind, datasets, settings), 0., 1.))] for ind in tqdm(population)]


def main(npop: int, ngen: int, num_samples: int=200, num_reps: int=1,
         save: bool=False):

    # -- settings
    settings = load_autoencoder(index=0)
    settings["num_samples"] = num_samples
    datasets = make_datasets(num_samples=num_samples, num_reps=num_reps,
                             dim_ei=settings["dim_ei"], K=int(settings["K"]))

    # -- save
    name = f"run_{len(os.listdir('logs/'))}"
    name += f"_{time.localtime().tm_mday}{time.localtime().tm_mon}"
    name += f"_{time.localtime().tm_hour}{time.localtime().tm_min}"
    info = {"genome": [],
            "fitness": 0.,
            "dim_ei": settings["dim_ei"],
            "dim_ca1": settings["dim_ca1"],
            "dim_eo": settings["dim_eo"],
            "K": settings["K"]}

    # -- evolution setup
    dim = 5
    num_lineages = 2
    size_lineage = npop
    spawn_rate = 30*6

    settings_ev = ev.EvolutionSettings(num_lineages, size_lineage, spawn_rate, 0.8);

    # genes
    space = np.zeros((dim, 2));
    for i in range(dim):
        space[i] = [-0.5, 0.5]

    space = [[0.5, 0.5]] * 3
    space += [[0.5, 0.5], [0.4, 0.4]]

    evolution = ev.Evolution(settings_ev, space);

    # plot
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots(dim, num_lineages, sharex=True, sharey=True)
    record = {"fitness": []}
    for _d in range(dim): record[_d] = []

    logger(f"NPOP={npop}")
    logger(f"NGEN={ngen}")
    logger(f"NLINEAGES={num_lineages}")
    logger(f"NREPS={num_reps}")
    logger(f"DIM={dim}")
    logger(f"NSTIMULI={num_samples}")
    logger(f"{save=}")
    logger("-----------------")

    # -- pre-run
    population = evolution.get_population()
    fitness = fit_population(population, datasets, settings)
    _fitness = -1.*np.log(np.array(fitness))
    logger(f"gen=0 | fitness={abs(np.min(fitness)):.3f}")

    # logs
    record = {}
    for l in range(num_lineages):
        record[l] = {"color": COLORS[l]}
        record[l]["fitness"] = [fitness]
        record[l]["history"] = [np.mean(fitness).item()]
        for _d in range(dim):
            _pop = []
            record[l][_d] = []
            for _ind in population:
                _pop += [_ind[_d]]
            record[l][_d] += [_pop]

    # -- run
    for gen in range(ngen):

        for lin in range(num_lineages):
            population = evolution.update(fitness)
            fitness = fit_population(population, datasets, settings)
            _fitness = -1.*np.log(np.array(fitness))
            logger(f"gen={gen+1} | [{lin}] fitness={np.max(_fitness):.3f}")

            # logs
            record[lin]["fitness"] += [_fitness]
            for _d in range(dim):
                _pop = []
                for _ind in population:
                    _pop += [_ind[_d]]
                record[lin][_d] += [_pop]

            record[lin]["history"] += [np.mean(_fitness.flatten()).item()]

            # plot
            ax.clear()
            ax.grid()
            for h in range(num_lineages):
                for x, g in enumerate(np.array(record[h]["fitness"])):
                    ax.scatter([x]*len(population), g, s=30, color=record[h]["color"],
                               marker="x", alpha=0.5)
                ax.plot(range(len(record[h]["history"])), record[h]["history"], color=record[h]["color"], lw=2)
            ax.set_xlabel("generations")
            ax.set_ylabel("fitness")

            for k in range(dim):
                for h in range(num_lineages):
                    ax2[k][h].clear()
                    ax2[k][h].grid()
                    for x, g in enumerate(np.array(record[h][k])):
                        ax2[k][h].scatter([x]*len(g), g, s=30, c=record[h]['fitness'][x],
                                       cmap="Blues_r", alpha=0.8)
                    ax2[k][h].set_ylabel(f"gene {k}")
                    if k == 0:
                        ax2[k][h].set_title(f"lineage {h}")

                # ax2[k][h].set_xlabel("generations")

            plt.pause(0.001)

        # save
        if save:
            info["genome"] = population[np.argmax(_fitness)]
            info["fitness"] = np.max(_fitness)
            save_genome(info=info, name=name)


# --------------------------------------------------
# --------------------------------------------------


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="evolution of the BTSP model")
    parser.add_argument('--npop', type=int,
                        help='population size',
                        default=8)
    parser.add_argument('--ngen', type=int,
                        help='number of generations',
                        default=100)
    parser.add_argument('--save', action='store_true',
                        help='session save option')
    parser.add_argument('--samples', type=int,
                        help='number of samples',
                        default=200)
    parser.add_argument('--reps', type=int,
                        help='number of repetitions',
                        default=1)
    args = parser.parse_args()

    main(npop=args.npop, ngen=args.ngen, num_samples=int(args.samples),
         num_reps=args.reps, save=args.save)

    logger("[done]")


