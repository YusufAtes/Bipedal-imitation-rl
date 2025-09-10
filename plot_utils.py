import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def set_pub_style():
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
    })

def edges_from_centers(vals):
    """Convert 1D bin centers → bin edges (works with non-uniform spacing)."""
    v = np.asarray(vals, dtype=float)
    if len(v) == 1:
        step = 1.0
        return np.array([v[0] - step/2, v[0] + step/2])
    mid = 0.5 * (v[:-1] + v[1:])
    left = v[0] - (v[1] - v[0]) / 2
    right = v[-1] + (v[-1] - v[-2]) / 2
    return np.r_[left, mid, right]

def plot_mse_heatmap(x_centers, y_centers, Z, xtitle, ytitle,title,savepath,
                     vmin=0.0, vmax=0.01,
                     cmap="turbo",        # MATLAB-like look; use "viridis" if you prefer
                     title_prefix="Validation Dataset Average MSE Value"):
    """
    x_centers: 1D array of Speed bins (e.g., [2,2.5,...,11])
    y_centers: 1D array of Ramp bins  (e.g., [-8,-7.5,...,8])
    Z        : 2D array (len(y_centers), len(x_centers)) with MSE values
    """
    set_pub_style()
    x_edges = edges_from_centers(x_centers)
    y_edges = edges_from_centers(y_centers)

    # If you want to hide invalid cells:
    # Z = np.ma.masked_invalid(Z)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))  # wide, like your example
    m = ax.pcolormesh(x_edges, y_edges, Z, shading="auto",
                      cmap=cmap, vmin=vmin, vmax=vmax,
                      edgecolors="k", linewidth=0.35)   # <- black grid lines

    # Colorbar
    cbar = fig.colorbar(m, ax=ax, pad=0.012)
    cbar.set_label("Distance (m)")

    # Labels
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)

    ax.set_title(title, pad=6)

    # Tick density & layout
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(10))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(8))
    fig.tight_layout()
    fig.savefig(savepath)
    return fig, ax