import torch
import torch.nn as nn
import numpy as np
import os


def plot_2d(input_1, foldername, filename):
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))

    # vmax, vmin = 8, -8
    pcm = ax[0].contourf(input_1.squeeze().numpy(),
                         cmap='jet', shading='auto')
    fig.colorbar(pcm, ax=ax[0], extend='max')
    ax[0].set_aspect("equal")

    # pcm = ax[1].contourf(input_2.squeeze().numpy(),
    #                      cmap='jet', shading='auto', vmin=vmin, vmax=vmax)
    # fig.colorbar(pcm, ax=ax[1], extend='max')
    # ax[1].set_aspect("equal")

    plt.savefig(os.path.join(foldername, filename),
                dpi=500, bbox_inches='tight')
