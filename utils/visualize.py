# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl
mpl.use('Agg')

def plot_scatter(pt, data_name, plt_path):
    fig = plt.figure()
    fig.set_size_inches(20.0 / 3, 20.0 / 3)
    ax = fig.gca(projection='3d')
    ax.set_aspect('auto')
    ax.grid(color='r', linestyle='-',)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])
    X = pt[:, 0]
    Y = pt[:, 1]
    Z = pt[:, 2]
    X_max = max(X)
    X_min = min(X)
    Y_max = max(Y)
    Y_min = min(Y)
    Z_max = max(Z)
    Z_min = min(Z)
    scat = ax.scatter(X, Y, Z, depthshade=True, marker='.')

    #max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
    max_range = np.array([X_max - X_min, Y_max - Y_min, Z_max - Z_min]).max() / 2.0

    mid_x = (X_max + X_min) * 0.5
    mid_y = (Y_max + Y_min) * 0.5
    mid_z = (Z_max + Z_min) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.margins(0)
    fig.savefig(os.path.join(plt_path, data_name.replace('.dat', '.png')), format='png', transparent=True, dpi=300, pad_inches=0, bbox_inches='tight')
