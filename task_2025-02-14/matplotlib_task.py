import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.default": "regular"})

import numpy as np


Nx = 100
Ny = 100
a = 1
b = 2

xmesh = np.linspace(-np.pi, np.pi, Nx)
ymesh = np.linspace(-np.pi, np.pi, Ny)

z = np.sin(b * ymesh).reshape(-1, 1) @ np.sin(a * xmesh).reshape(1, -1)
u1 = z + np.random.normal(loc=0., scale=0.1, size=z.shape)
u2 = z + np.random.poisson(lam=0.01, size=z.shape)

mosaic = \
"""
zzuudd
zzuudd
ZZUUDD
ZZUUDD
"""

cmap="gist_rainbow_r"

beautiful_labels = [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"]
plots_config = [
    ("z", z, "a)", [], []),
    ("u", u1, "b)", beautiful_labels, beautiful_labels),
    ("d", np.abs(z - u1), "c)", [], []),
    ("Z", z, "d)", [], []),
    ("U", u2, "e)", [], []),
    ("D", np.abs(z - u2), "f)", [], []),
]
fig, ax = plt.subplot_mosaic(mosaic,figsize=(10, 8), constrained_layout=True)

for i, cfg in enumerate(plots_config):
    ax_code, data, title, xlabels, ylabels = cfg
    img = ax[ax_code].imshow(data[:, ::-1], cmap=cmap)
    plots_config[i] = cfg + (img, )
    ax[ax_code].set_title(title)
    ax[ax_code].set_xlabel("x")
    ax[ax_code].set_ylabel("y")
    if len(xlabels) > 0:
        ax[ax_code].set_xticks(np.linspace(0, Nx, len(xlabels)))
        ax[ax_code].set_xticklabels(xlabels)
    else:
        ax[ax_code].set_xticks([])
    if len(ylabels) > 0:
        ax[ax_code].set_yticks(np.linspace(0, Ny, len(ylabels))[::-1])
        ax[ax_code].set_yticklabels(ylabels)
    else:
        ax[ax_code].set_yticks([])


cbar_u = plt.colorbar(plots_config[0][5], ax=(ax["u"], ax["U"]), shrink=0.8)
cbar_d = plt.colorbar(plots_config[2][5], ax=ax["d"], shrink=0.6)
cbar_D = plt.colorbar(plots_config[5][5], ax=ax["D"], shrink=0.6)
for cbar in (cbar_u, cbar_d, cbar_D):
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)

fig.savefig("output.eps")
fig.savefig("output.png")
plt.show()
