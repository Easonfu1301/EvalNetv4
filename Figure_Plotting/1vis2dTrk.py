import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from tqdm import tqdm
from utils import *


def vis_track_p(df, p_cut: float = 1500) -> None:
    """Visualise the x–y projection of tracks coloured by momentum.

    Replaces the ``LogNorm`` colour‐scale with a ``1 + \log_{10}`` transformation
    so the colour bar still reflects the *actual* momentum (MeV/c) values but the
    mapping is handled explicitly rather than via ``matplotlib.colors.LogNorm``.
    """
    # Keep only the first event and tracks above the requested p cut
    df = df[(df["eventID"] == 1) & (df["p"] > p_cut)]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Momentum range for normalisation (after 1+log10 transformation)
    max_p = df["p"].max()
    min_p = df["p"].min()
    log_min = np.log10(p_cut + 1)
    log_max = np.log10(max_p + 1)
    norm = mcolors.Normalize(vmin=log_min, vmax=log_max)  # linear norm on the transformed scale

    # Draw each MC‐particle track
    for pid in tqdm(df["mcparticleID"].unique(), desc="Drawing tracks"):
        track = df[df["mcparticleID"] == pid]
        xyz = track[["x", "y", "z"]].values
        p_vals = track["p"].values
        # colour based on 1+log10(p) so that very high-momentum tracks stand out
        colours = plt.cm.jet(norm(np.log10(p_vals + 1)))
        ax.plot(
            xyz[:, 0],
            xyz[:, 1],
            ".-",
            c=colours[0],  # use the colour of the first point for the whole track
            alpha=0.5,
        )

    # Axes styling
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_xlim(-750, 750)
    ax.set_ylim(-750, 750)

    # Colour-bar with real p values
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical")
    cbar.set_label("p (MeV/c)")

    # Use 6 log-spaced ticks and label them with the actual numbers
    print(f"Min p: {min_p}, Max p: {max_p}")
    tick_values = np.geomspace(min_p, max_p, num=6)
    cbar.set_ticks(np.log10(tick_values + 1))  # positions on the transformed scale
    cbar.set_ticklabels([_sci_label(v) for v in tick_values])

    plt.title(f"Orthogonal projection of tracks with p > {p_cut} MeV/c")
    ax.set_aspect('equal', adjustable='box')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.minorticks_on()
    # set inner minor ticks
    ax.tick_params(which='major', direction='in', length=6, width=1.5, colors='black', grid_color='gray', grid_alpha=0.5)
    ax.tick_params(which='minor', direction='in', length=3, width=1, colors='black', grid_color='gray', grid_alpha=0.5)

    # set frame width
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()

    fig.savefig(os.path.join(workdir, "Plot", f"vis_track_p_cut_{p_cut}.png"), dpi=300)

    plt.show()


def _sci_label(value: float) -> str:
    """Return a LaTeX math-formatted *n×10^x* string for *value*."""
    if value == 0:
        return "$0$"  # safeguard; shouldn't appear in log scale
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / 10 ** exponent
    # Correct rounding artefacts (e.g. 9.99999 → 10)
    if abs(mantissa - 10) < 1e-6:
        mantissa = 1
        exponent += 1
    return f"${mantissa:.2f}\\times10^{{{exponent}}}$"


def main() -> None:
    raw_dir = os.path.join(workdir, "RawData")
    raw_files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
    if not raw_files:
        raise FileNotFoundError("No CSV files found in RawData/")

    df = pd.read_csv(os.path.join(raw_dir, raw_files[0]))
    vis_track_p(df, p_cut=0)
    vis_track_p(df, p_cut=1500)


if __name__ == "__main__":
    main()
