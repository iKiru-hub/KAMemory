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

if "doki" in os.getcwd():
    sys.path.append(os.getcwd().split("doki")[0] + \
        "doki/side_lab/evolution_cpp/core/cpp1/build")
    import evolution as ev
elif "Research" in os.getcwd():
    sys.path.append(os.getcwd().split("Research")[0] + \
        "Research/studio/evolution_cpp/core/cpp1/build")
    import evolution as ev
else:
    raise ModuleNotFoundError("not found at all")


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
GENOME_KEYS = ("K_lat", "K_ca3", "K_out", "beta", "alpha")
SIGMA = 30

GENOME_CONFIGS = {
    "K_lat" : {"active": False,
               "color": "red",
               "init": 15,
               "var": 5,
               "min": 10,
               "max": 100},
    "K_out" : {"active": False,
               "color": "blue",
               "init": 5,
               "var": 1,
               "min": 10,
               "max": 100},
    "K_ca3" : {"active": True,
               "color": "black",
               "init": 15,
               "var": 5,
               "scale": 10,
               "min": 1,
               "max": 50},
    "beta_eo"  : {"active": True,
                  "color": "green",
                  "init": 20,
                  "var": 8,
                  "scale": 100,
                  "min": 0,
                  "max": 400},
    "beta_is"  : {"active": True,
                  "color": "green",
                  "init": 20.,
                  "var": 8.,
                  "scale": 100,
                  "min": 0,
                  "max": 400},
    "beta_ca1"  : {"active": True,
                   "color": "green",
                   "init": 100,
                   "var": 10,
                   "scale": 100,
                   "min": 0,
                   "max": 400},
    "beta_ca3"  : {"active": True,
                   "color": "green",
                   "init": 100,
                   "var": 10,
                   "scale": 100,
                   "min": 0,
                   "max": 400},
    "alpha" : {"active": True,
               "color": "orange",
               "init": 0.1,
               "var": 0.05,
               "scale": 1.,
               "min": 0,
               "max": 1},
    "num_swaps_ca1": {"active": False,
                      "color": "purple",
                      "init": 1.,
                      "var": 2.,
                      "scale": 1.,
                      "min": 1,
                      "max": 50},
    "num_swaps_ca3": {"active": False,
                      "color": "purple",
                      "init": 1.,
                      "var": 2.,
                      "scale": 1.,
                      "min": 1,
                      "max": 10}
}

# --------------------------------------------------
# --------------------------------------------------

def fit_population(population: list, datasets: list, settings: dict):
    # return [[ np.exp(-1. * np.clip(evaluate_genome(ind, datasets, settings), 0., 1.))] for ind in tqdm(population)]

    fitted = []
    for ind in population:
        try:
            fitted += [[ np.exp(-1. * np.clip(evaluate_genome(ind, datasets, settings),
                                              0., 1.))]]
        except Exception:
            fitted += [[0.]]

    return fitted


def main(npop: int, ngen: int, num_samples: int=200, num_reps: int=1,
         save: bool=False, load_index: int=0, K_data: int=-1,
         num_lineages: int=2):

    # -- settings
    settings = load_autoencoder(index=load_index)
    settings["num_samples"] = num_samples
    settings["sigma"] = SIGMA
    settings["genome_configs"] = GENOME_CONFIGS
    # for k, v in settings["genome_configs"].items():
    #     settings["genome_configs"][k]["var"] = np.sqrt(
    #             settings["genome_configs"][k]["max"] - \
    #             settings["genome_configs"][k]["min"])

    # -- dataset
    if K_data == -1:
        K_data = int(settings["K_lat"])
    datasets = make_datasets(num_samples=num_samples, num_reps=num_reps,
                             dim_ei=settings["dim_ei"], K=K_data)

    # -- save
    name = f"run_{len(os.listdir('logs/'))}"
    name += f"_{time.localtime().tm_mday}{time.localtime().tm_mon}"
    name += f"_{time.localtime().tm_hour}{time.localtime().tm_min}"
    info = {"genome": {},
            "fitness": 0.,
            "genome_configs": GENOME_CONFIGS,
            "dim_ei": settings["dim_ei"],
            "dim_ca1": settings["dim_ca1"],
            "dim_eo": settings["dim_eo"],
            "dim_ca3": settings["genome_configs"]["K_ca3"]["max"],
            "K": settings["K_lat"],
            "sigma": SIGMA,
            "notes": "idk hope it works"}

    # -- evolution setup
    dim = 0
    gene_names = []
    gene_idx = []
    for i, (k, v) in enumerate(GENOME_CONFIGS.items()):
        if v["active"]:
            gene_names += [k]
            gene_idx += [i]
            dim += 1
    size_lineage = npop
    spawn_rate = -1

    settings_ev = ev.EvolutionSettings(num_lineages, size_lineage, spawn_rate, 0.8)

    # genes
    # space = np.zeros((dim, 2));
    # for i in range(dim):
    #     space[i] = [0.1, 0.4]

    means = []
    variances = []
    for k, v in settings["genome_configs"].items():
        means += [float(v["init"])]
        variances += [float(v["var"])]

    # space = [[0.5, 0.5]] * 3
    # space += [[0.5, 0.5], [0.4, 0.4]]

    logger.debug(f"{variances=}")
    evolution = ev.Evolution(settings_ev, means, variances)

    # plot
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots(dim, num_lineages, sharex=True, sharey=False)
    record = {"fitness": [], "sigma": SIGMA}
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
    save_record = {}
    for l in range(num_lineages):
        record[l] = {"color": COLORS[l]}
        record[l]["fitness"] = [fitness]
        record[l]["history"] = [np.mean(fitness).item()]
        save_record[l] = {}
        save_record[l]["fitness"] = [_fitness.tolist()]
        for _d in range(dim):
            _pop = []
            record[l][_d] = []
            save_record[l][_d] = []
            for _ind in population:
                _pop += [_ind[gene_idx[_d]]]
            record[l][_d] += [_pop]
            save_record[l][_d] += [_pop]

    # -- run
    best_fitness = 0.
    best_genome = []
    for gen in range(ngen):

        for lin in range(num_lineages):
            population = evolution.update(fitness)

            fitness = fit_population(population, datasets, settings)
            _fitness = -1.*np.log(np.array(fitness))
            logger(f"gen={gen+1} | [{lin}] fitness={np.max(_fitness):.3f}")

            # record best genome
            if best_fitness < _fitness.max():
                best_fitness = _fitness.max()
                best_genome = population[np.argmax(_fitness)]

            # logs
            record[lin]["fitness"] += [_fitness]
            for _d in range(dim):
                _pop = []
                for _ind in population:
                    _pop += [_ind[gene_idx[_d]]]
                record[lin][_d] += [_pop]
                save_record[lin][_d] += [_pop]

            record[lin]["history"] += [np.mean(_fitness.flatten()).item()]
            save_record[lin]["fitness"] += [_fitness.tolist()]

            # plot
            ax.clear()
            ax.grid()
            for h in range(num_lineages):
                for x, g in enumerate(np.array(record[h]["fitness"])):
                    ax.scatter([x]*len(population), g, s=30, color=record[h]["color"],
                               marker="x", alpha=0.3)
                ax.plot(range(1, len(record[h]["history"])), record[h]["history"][1:], color=record[h]["color"], lw=2,
                        label=f"lineage {h}")
            ax.set_xlabel("generations")
            ax.set_ylabel("fitness")
            ax.legend(loc="upper left")
            ax.set_ylim((0, 1))
            ax.set_xlim((1, len(record[0]["history"])))

            # for k in range(dim):
            for k in range(dim):
                if len(record) == 0: break
                for h in range(num_lineages):
                    ax2[k][h].clear()
                    ax2[k][h].grid()
                    # for x, g in enumerate(np.array(record[h][k])):
                        # ax2[k][h].scatter([x]*len(g), g, s=30, c=record[h]['fitness'][x],
                        #                cmap="Blues_r", alpha=0.8)
                        # ax2[k][h].scatter([x], g.mean(), s=40, color="blue")
                        # ax2[k][h].plot([x, x], [g.mean()+g.var(), g.mean()-g.var()], "b-")

                    x_range = range(len(record[h][k]))
                    mean = np.array(record[h][k]).mean(axis=1)
                    std = np.array(record[h][k]).std(axis=1)
                    ax2[k][h].fill_between(x_range, mean - std, mean + std, alpha=0.2, color="blue")
                    ax2[k][h].plot(x_range, mean, "-|", color="blue", alpha=0.8)
                    # ax2[k][h].set_ylim(minv, maxv)
                    # ax2[k][h].set_ylabel(f"gene {k}")
                    if h == 0: ax2[k][h].set_ylabel(f"{gene_names[k]}")
                    if k == 0: ax2[k][h].set_title(f"lineage {h}")

                # ax2[k][h].set_xlabel("generations")

            plt.pause(0.001)

        # save
        if save:
            info["genome"] = {_n: _g for _n, _g in zip(gene_names, best_genome)}
            info["fitness"] = best_fitness
            # info["record"] = record
            info["record"] = save_record
            save_genome(info=info, name=name)


# --------------------------------------------------
# --------------------------------------------------


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="evolution of the BTSP model")
    parser.add_argument('--pop', type=int,
                        help='population size',
                        default=8)
    parser.add_argument('--gen', type=int,
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
    parser.add_argument('--index', type=int,
                        help='index of the autoencoder',
                        default=1)
    parser.add_argument('--lineages', type=int,
                        help='number of lineages',
                        default=2)
    args = parser.parse_args()


    main(npop=args.pop, ngen=args.gen, num_samples=int(args.samples),
         num_reps=args.reps, save=args.save,
         load_index=args.index,
         num_lineages=args.lineages)

    logger("[done]")


