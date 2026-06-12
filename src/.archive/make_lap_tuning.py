import numpy as np
import matplotlib.pyplot as plt
import os, argparse

import models
import utils
import main

logger = models.logger
logger(".")

parser = argparse.ArgumentParser(
    description="reconstruction plot")
parser.add_argument('--idx', type=int,
                    help='number of samples',
                    default=None)
args = parser.parse_args()

results = utils.train_for_reconstruction(alpha=0.3, num_samples=5, verbose=False, bias=False, use_lec=True, idx=args.idx, binarize=False)

""" plot """
# fig2, (ax12, ax22, ax32) = plt.subplots(3, 1,
#                                 figsize=(10, 6), sharex=True)

fig2, (ax12, ax22, ax32, ax42) = plt.subplots(4, 1,
                                figsize=(10, 8), sharex=True)

plt.subplots_adjust(hspace=0.3)

is_squash = False

utils.plot_squashed_data(data=results["data"],
                         ax=ax12,
                         title="Patterns",
                         proper_title=True)

utils.plot_squashed_data(data=results["out_ae"], ax=ax22,
                         title=f"$AE$",
                         proper_title=True)

utils.plot_squashed_data(data=results["out_mtl"], ax=ax32,
                         title=f"$IS$ - " + \
                f"reconstruction loss={results['rec_loss']:.3f}",
                         proper_title=True)
utils.plot_squashed_data(data=results["out_mtl_rnd"], ax=ax42,
                         title=f"shuffled $IS$ - " + \
                f"reconstruction loss={results['rec_loss_rnd']:.3f}",
                         proper_title=True)


plt.show()
