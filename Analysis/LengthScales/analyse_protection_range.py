"""
analyse_protection_range.py
============================
Analyses CellModeller pickle files from anti_circle_island.py to determine
the spatial length scale over which resistant cells protect sensitive cells
from antibiotic.

Methods implemented
-------------------
1. Growth-rate decay profile
   For each sensitive cell, compute its distance to the nearest resistant
   cell (or to the island edge).  Bin sensitive cells by distance and plot
   mean growth rate vs. distance.  The protection range λ is the distance
   at which growth rate falls to half its maximum (analogous to the "growth
   range" in Dal Co et al. 2020, Fig. 4a).

2. Spearman correlation method  (Dal Co et al. Fig. 2d equivalent)
   For each sensitive cell, compute the fraction of resistant cells within a
   neighbourhood of radius d.  Correlate that fraction with the cell's growth
   rate across all sensitive cells.  The peak of ρ(d) is the protection range.

3. Time series  (λ vs. simulation step)
   Both methods are applied to every loaded pickle so you can track how the
   protection range grows as resistant cells deplete more antibiotic.

4. Colony-size comparison
   Point the script at multiple run folders (different ISLAND_RADIUS values)
   and it overlays the results.

Usage
-----
# Single run, all pickles in a folder:
    python analyse_protection_range.py --runs path/to/run_r20

# Multiple runs for colony-size comparison:
    python analyse_protection_range.py --runs run_r10 run_r20 run_r40 \
                                       --labels "r=10" "r=20" "r=40"

# Limit which pickle steps to load (saves time):
    python analyse_protection_range.py --runs run_r20 --steps 1 5 10 20 50

Dependencies: numpy, scipy, matplotlib, pickle (stdlib)
"""

import os
import glob
import pickle
import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from scipy.stats import spearmanr
from scipy.optimize import curve_fit

# ── Parameters you may want to tweak ─────────────────────────────────────────

# Range of neighbourhood radii to sweep for the Spearman method (µm)
D_MIN   = 1.0
D_MAX   = 120.0
N_D     = 60

# Minimum number of sensitive cells required to attempt an analysis
MIN_SENSITIVE = 30

# Physical parameters from the simulation (used for analytical prediction)
Diff_A      = 2400.0   # extracellular diffusion constant
k_A         = 10.0     # intracellular degradation rate (resistant cells)
D_membrane  = 20.0     # membrane permeability (D_A in the CL kernel)

# Analytical steady-state prediction: λ = sqrt(Diff_A / k_A)
# Note: this is the coarse 1-D slab estimate; true value depends on geometry
LAMBDA_ANALYTICAL = np.sqrt(Diff_A / k_A)

# ── I/O helpers ───────────────────────────────────────────────────────────────

def find_pickles(folder):
    """Return sorted list of .pickle files in a CellModeller output folder."""
    pattern = os.path.join(folder, "*.pickle")
    files   = sorted(glob.glob(pattern))
    if not files:
        # Some versions write step_NNNN.pkl
        pattern = os.path.join(folder, "*.pkl")
        files   = sorted(glob.glob(pattern))
    return files


def load_cells(pickle_path):
    """
    Load a CellModeller pickle and return the cellStates dict.
    CellModeller pickles are either the raw dict or a SimState object.
    """
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    # CellModeller ≥4 wraps everything in a dict with key 'cellStates'
    if isinstance(data, dict):
        return data.get("cellStates", data)
    # Older versions store the dict directly or as an attribute
    if hasattr(data, "cellStates"):
        return data.cellStates
    return data


def step_number(pickle_path):
    """Extract the integer step number from the filename."""
    base = os.path.splitext(os.path.basename(pickle_path))[0]
    digits = "".join(c for c in base if c.isdigit())
    return int(digits) if digits else 0


# ── Cell-state extraction ─────────────────────────────────────────────────────

def extract_arrays(cells):
    """
    Pull positions, cell types, growth rates, and intracellular antibiotic
    out of the cellStates dict into plain numpy arrays.

    Returns
    -------
    pos   : (N, 2) float   – x, y positions in µm
    ctype : (N,)   int     – 0 = sensitive, 1 = resistant
    gr    : (N,)   float   – current growth rate
    A_in  : (N,)   float   – intracellular antibiotic concentration
    """
    ids = list(cells.keys())
    pos   = np.array([cells[i].pos[:2]      for i in ids], dtype=float)
    ctype = np.array([cells[i].cellType     for i in ids], dtype=int)
    gr    = np.array([cells[i].growthRate   for i in ids], dtype=float)
    ends  = np.array([
        [cells[i].ends[0][:2], cells[i].ends[1][:2]]
        for i in ids
    ], dtype=float)
    radii = np.array([cells[i].radius for i in ids], dtype=float)

    # species[0] is intracellular antibiotic in our model
    try:
        A_in = np.array([cells[i].species[0] for i in ids], dtype=float)
    except Exception:
        A_in = np.zeros(len(ids))

    return pos, ctype, gr, A_in, ends, radii


# ── Method 1: distance-decay profile ─────────────────────────────────────────

def nearest_resistant_distances(pos, ctype):
    """
    For each sensitive cell, find the distance to the nearest resistant cell.

    Returns
    -------
    dists : (N_sens,) float   distances in µm
    idx_s : (N_sens,) int     indices into pos/ctype arrays
    """
    idx_s = np.where(ctype == 0)[0]
    idx_r = np.where(ctype == 1)[0]

    if len(idx_s) == 0 or len(idx_r) == 0:
        return np.array([]), np.array([], dtype=int)

    sens_pos = pos[idx_s]   # (N_s, 2)
    res_pos  = pos[idx_r]   # (N_r, 2)

    # Vectorised pairwise distances: (N_s, N_r)
    diff  = sens_pos[:, None, :] - res_pos[None, :, :]   # (N_s, N_r, 2)
    dists = np.min(np.linalg.norm(diff, axis=2), axis=1)  # (N_s,)
    return dists, idx_s


def decay_profile(pos, ctype, gr, n_bins=20):
    """
    Bin sensitive cells by distance to nearest resistant cell.
    Return bin centres and mean growth rate per bin.
    """
    dists, idx_s = nearest_resistant_distances(pos, ctype)
    if len(dists) == 0:
        return None, None, None

    gr_s = gr[idx_s]

    d_max = np.percentile(dists, 95)   # ignore extreme outliers
    edges = np.linspace(0, d_max, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    mean_gr  = np.full(n_bins, np.nan)
    sem_gr   = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = (dists >= edges[i]) & (dists < edges[i + 1])
        if mask.sum() >= 3:
            mean_gr[i] = np.mean(gr_s[mask])
            sem_gr[i]  = np.std(gr_s[mask]) / np.sqrt(mask.sum())

    return centres, mean_gr, sem_gr


def half_max_distance(centres, mean_gr):
    """
    Estimate the distance at which growth rate falls to half its maximum,
    using linear interpolation between bins.
    Returns NaN if it cannot be determined.
    """
    valid = ~np.isnan(mean_gr)
    if valid.sum() < 2:
        return np.nan

    gr_max = np.nanmax(mean_gr)
    half   = gr_max / 2.0

    c = centres[valid]
    g = mean_gr[valid]

    # Find first crossing below half-max
    for i in range(len(g) - 1):
        if g[i] >= half and g[i + 1] < half:
            # Linear interpolation
            frac = (g[i] - half) / (g[i] - g[i + 1])
            return c[i] + frac * (c[i + 1] - c[i])
    return np.nan


# ── Method 2: Spearman correlation (Dal Co method) ───────────────────────────

def resistant_fraction_in_radius(pos, ctype, d):
    """
    For every sensitive cell, compute the fraction of ALL cells within radius d
    that are resistant.

    Returns fractions array aligned with sensitive-cell indices.
    """
    idx_s = np.where(ctype == 0)[0]
    if len(idx_s) == 0:
        return np.array([])

    fractions = np.empty(len(idx_s))
    for k, i in enumerate(idx_s):
        dists_to_all = np.linalg.norm(pos - pos[i], axis=1)
        in_range     = dists_to_all < d
        n_in         = in_range.sum()
        if n_in == 0:
            fractions[k] = 0.0
        else:
            fractions[k] = (ctype[in_range] == 1).sum() / n_in

    return fractions


def spearman_curve(pos, ctype, gr, d_values=None):
    """
    Compute Spearman ρ between resistant fraction and sensitive cell growth
    rate for each neighbourhood radius d.

    Returns
    -------
    d_values   : (N_d,) array of radii tested
    rho_values : (N_d,) Spearman correlation coefficients
    peak_d     : float  radius at which ρ is maximal  (= protection range)
    """
    if d_values is None:
        d_values = np.linspace(D_MIN, D_MAX, N_D)

    idx_s = np.where(ctype == 0)[0]
    if len(idx_s) < MIN_SENSITIVE:
        return d_values, np.full(len(d_values), np.nan), np.nan

    gr_s = gr[idx_s]

    # Skip if growth rates show no variation (all cells equally suppressed)
    if np.std(gr_s) < 1e-6:
        return d_values, np.full(len(d_values), np.nan), np.nan

    rho_values = np.empty(len(d_values))
    for j, d in enumerate(d_values):
        fracs = resistant_fraction_in_radius(pos, ctype, d)
        if np.std(fracs) < 1e-9:
            rho_values[j] = np.nan
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho, _ = spearmanr(fracs, gr_s)
            rho_values[j] = rho

    valid      = ~np.isnan(rho_values)
    if valid.sum() == 0:
        return d_values, rho_values, np.nan

    peak_idx = np.nanargmax(rho_values)
    peak_d   = d_values[peak_idx]
    return d_values, rho_values, peak_d


# ── Single-timepoint analysis ─────────────────────────────────────────────────

def analyse_timepoint(pickle_path, d_values=None, n_bins=20, verbose=True):
    """
    Run both analyses on a single pickle file.

    Returns a result dict with keys:
        step, n_sensitive, n_resistant,
        decay_centres, decay_mean_gr, decay_sem_gr,
        lambda_half_max,
        d_values, rho_values, lambda_spearman
    """
    cells = load_cells(pickle_path)
    step  = step_number(pickle_path)

    pos, ctype, gr, A_in, ends, radii = extract_arrays(cells)
    n_s = (ctype == 0).sum()
    n_r = (ctype == 1).sum()

    if verbose:
        print(f"  Step {step:4d}  |  sensitive={n_s}  resistant={n_r}")

    # Method 1
    centres, mean_gr, sem_gr = decay_profile(pos, ctype, gr, n_bins=n_bins)
    lam_hm = half_max_distance(centres, mean_gr) if centres is not None else np.nan

    # Method 2
    if d_values is None:
        d_values = np.linspace(D_MIN, D_MAX, N_D)
    d_vals, rho_vals, lam_sp = spearman_curve(pos, ctype, gr, d_values)

    return dict(
        step           = step,
        n_sensitive    = n_s,
        n_resistant    = n_r,
        decay_centres  = centres,
        decay_mean_gr  = mean_gr,
        decay_sem_gr   = sem_gr,
        lambda_half_max= lam_hm,
        d_values       = d_vals,
        rho_values     = rho_vals,
        lambda_spearman= lam_sp,
        pos            = pos,
        ctype          = ctype,
        gr             = gr,
        A_in           = A_in,
        ends           = ends,
        radii          = radii,
    )


# ── Full run analysis ─────────────────────────────────────────────────────────

def analyse_run(folder, steps_to_load=None, label="run"):
    """
    Analyse all (or selected) pickles in a run folder.

    Parameters
    ----------
    folder        : path to CellModeller output directory
    steps_to_load : list of int step numbers to load, or None for all
    label         : string label for plots

    Returns list of result dicts (one per loaded pickle).
    """
    pickles = find_pickles(folder)
    if not pickles:
        raise FileNotFoundError(f"No pickle files found in {folder}")

    if steps_to_load is not None:
        target = set(steps_to_load)
        pickles = [p for p in pickles if step_number(p) in target]

    print(f"\n── Analysing run '{label}'  ({len(pickles)} files) ──")
    d_values = np.linspace(D_MIN, D_MAX, N_D)

    results = []
    for p in pickles:
        res = analyse_timepoint(p, d_values=d_values)
        results.append(res)

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_spearman_curves(results, label="", ax=None, cmap_name="plasma"):
    """
    Plot ρ(d) curves for multiple timepoints on one axis,
    colour-coded by simulation step.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    steps = np.array([r["step"] for r in results])
    cmap_name = "tab10" if len(results) <= 10 else "turbo"
    colours = plt.get_cmap(cmap_name)(np.linspace(0, 1, len(results)))

    for r, col in zip(results, colours):
        rho = r["rho_values"]
        if np.all(np.isnan(rho)):
            continue
        ax.plot(r["d_values"], rho, color=col, alpha=0.8, lw=1.2,
                label=f"step {r['step']}")

    #ax.axvline(LAMBDA_ANALYTICAL, color="red", ls="--", lw=1.5,
    #           label=f"analytical λ = {LAMBDA_ANALYTICAL:.1f} µm")
    ax.set_xlabel("Neighbourhood radius d (µm)")
    ax.set_ylabel("Spearman ρ  (resistant fraction vs. growth rate)")
    ax.set_title(f"Spearman correlation curves — {label}")
    ax.legend(fontsize=6, ncol=2)
    return ax


def plot_decay_profiles(results, label="", steps_to_show=None, ax=None):
    """
    Plot growth-rate decay profiles for selected timepoints.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    if steps_to_show is None:
        to_show = [r for r in results if r["step"] != 0]
    else:
        step_set = set(steps_to_show)
        to_show  = [r for r in results if r["step"] in step_set]

    colours = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(to_show)))

    for r, col in zip(to_show, colours):
        c = r["decay_centres"]
        g = r["decay_mean_gr"]
        e = r["decay_sem_gr"]
        if c is None:
            continue
        valid = ~np.isnan(g)
        ax.plot(c[valid], g[valid], "-o", color=col,
                ms=4, lw=1.5, label=f"step {r['step']}")
        ax.fill_between(c[valid],
                        g[valid] - e[valid],
                        g[valid] + e[valid],
                        color=col, alpha=0.2)

    ax.set_xlabel("Distance to nearest resistant cell (µm)")
    ax.set_ylabel("Mean growth rate of sensitive cells")
    ax.set_title(f"Growth-rate decay profile — {label}")
    ax.legend(fontsize=8)
    return ax


def plot_lambda_vs_time(results_list, labels, ax=None):
    """
    Plot both λ estimates vs. simulation step for one or more runs.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    markers = ["o", "s", "^", "D", "v"]
    colours = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for i, (results, label) in enumerate(zip(results_list, labels)):
        steps    = np.array([r["step"]            for r in results])
        lam_hm   = np.array([r["lambda_half_max"] for r in results])
        lam_sp   = np.array([r["lambda_spearman"] for r in results])

        col = colours[i % len(colours)]
        mk  = markers[i % len(markers)]

        ax.plot(steps, lam_hm, f"-{mk}", color=col, alpha=0.8,
                label=f"{label} — half-max method")
        ax.plot(steps, lam_sp, f"--{mk}", color=col, alpha=0.5,
                label=f"{label} — Spearman method")

    #ax.axhline(LAMBDA_ANALYTICAL, color="red", ls=":", lw=2,
    #           label=f"analytical steady-state λ = {LAMBDA_ANALYTICAL:.1f} µm")
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Protection range λ (µm)")
    ax.set_title("Protection range over time")
    ax.legend(fontsize=8)
    return ax


def plot_colony_size_comparison(results_list, labels, target_step=-1, ax=None):
    """
    At a given timepoint, compare the Spearman ρ(d) curves for runs with
    different island radii.  target_step=-1 means the last available step.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    colours = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for i, (results, label) in enumerate(zip(results_list, labels)):
        if not results:
            continue
        r = results[target_step]
        rho = r["rho_values"]
        if np.all(np.isnan(rho)):
            continue
        ax.plot(r["d_values"], rho, color=colours[i % len(colours)],
                lw=2, label=f"{label}  (step {r['step']})")
        # Mark the peak
        peak_d = r["lambda_spearman"]
        if not np.isnan(peak_d):
            peak_rho = np.nanmax(rho)
            ax.axvline(peak_d, color=colours[i % len(colours)],
                       ls="--", alpha=0.5)
            ax.text(peak_d + 1, peak_rho * 0.95,
                    f"λ={peak_d:.1f}", fontsize=8,
                    color=colours[i % len(colours)])

    #ax.axvline(LAMBDA_ANALYTICAL, color="red", ls=":", lw=1.5,
    #           label=f"analytical λ={LAMBDA_ANALYTICAL:.1f} µm")
    ax.set_xlabel("Neighbourhood radius d (µm)")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Colony size comparison — Spearman curves")
    ax.legend(fontsize=8)
    return ax


def plot_spatial_snapshot(result, ax=None):
    """
    Plot capsule-shaped cells, coloured by growth rate, for one timepoint.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ctype = result["ctype"]
    gr    = result["gr"]
    ends  = result["ends"]
    radii = result["radii"]

    idx_s = ctype == 0
    idx_r = ctype == 1

    def capsule_patch(e1, e2, radius):
        direction = e2 - e1
        angle = np.arctan2(direction[1], direction[0])
        normal = np.array([-np.sin(angle), np.cos(angle)])
        end_angles = np.linspace(angle - np.pi / 2, angle + np.pi / 2, 13)
        start_angles = np.linspace(angle + np.pi / 2, angle + 3 * np.pi / 2, 13)
        end_arc = e2 + radius * np.column_stack(
            (np.cos(end_angles), np.sin(end_angles))
        )
        start_arc = e1 + radius * np.column_stack(
            (np.cos(start_angles), np.sin(start_angles))
        )
        vertices = np.vstack((
            e1 - radius * normal,
            e2 - radius * normal,
            end_arc,
            e1 + radius * normal,
            start_arc,
        ))
        codes = np.full(len(vertices), MplPath.LINETO, dtype=np.uint8)
        codes[0] = MplPath.MOVETO
        codes[-1] = MplPath.CLOSEPOLY
        return PathPatch(MplPath(vertices, codes))

    patches = [
        capsule_patch(ends[i, 0], ends[i, 1], radii[i])
        for i in range(len(ctype))
    ]
    sensitive = PatchCollection(
        [patches[i] for i in np.flatnonzero(idx_s)],
        cmap="turbo", edgecolor="black", linewidth=0.25, label="sensitive",
    )
    sensitive.set_array(gr[idx_s])
    sensitive.set_clim(0, 1.93)
    ax.add_collection(sensitive)

    resistant = PatchCollection(
        [patches[i] for i in np.flatnonzero(idx_r)],
        facecolor="violet", edgecolor="black", linewidth=0.25,
        alpha=0.7, label="resistant",
    )
    ax.add_collection(resistant)

    plt.colorbar(sensitive, ax=ax, label="growth rate")
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_title(f"Spatial snapshot — step {result['step']}")
    ax.legend(fontsize=8)
    return ax


# ── Main entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse protection length scale from CellModeller pickles"
    )
    parser.add_argument(
        "--runs", nargs="+", required=True,
        help="One or more run folders (output of CellModeller)"
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Labels for each run (default: folder names)"
    )
    parser.add_argument(
        "--steps", nargs="+", type=int, default=None,
        help="Which simulation steps to load (default: all)"
    )
    parser.add_argument(
        "--outdir", default="figures",
        help="Directory to save figures (default: ./figures)"
    )
    args = parser.parse_args()

    labels = args.labels if args.labels else args.runs
    os.makedirs(args.outdir, exist_ok=True)

    # ── Load all runs ──────────────────────────────────────────────────────
    all_results = []
    for folder, label in zip(args.runs, labels):
        results = analyse_run(folder, steps_to_load=args.steps, label=label)
        all_results.append(results)

    # ── Figure 1: Spatial snapshot at first and last timepoint ────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_spatial_snapshot(all_results[0][0],  ax=axes[0])
    axes[0].set_title(f"Early — step {all_results[0][0]['step']}")
    plot_spatial_snapshot(all_results[0][-1], ax=axes[1])
    axes[1].set_title(f"Late  — step {all_results[0][-1]['step']}")
    fig.suptitle(f"Spatial snapshots — {labels[0]}", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "fig1_snapshots.png"), dpi=150)
    print(f"\nSaved fig1_snapshots.png")

    # ── Figure 2: Growth-rate decay profiles over time (first run) ────────
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_decay_profiles(all_results[0], label=labels[0], ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "fig2_decay_profiles.png"), dpi=150)
    print("Saved fig2_decay_profiles.png")

    # ── Figure 3: Spearman ρ(d) curves over time (first run) ─────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_spearman_curves(all_results[0], label=labels[0], ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "fig3_spearman_curves.png"), dpi=150)
    print("Saved fig3_spearman_curves.png")

    # ── Figure 4: λ vs. time (all runs overlaid) ──────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_lambda_vs_time(all_results, labels, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "fig4_lambda_vs_time.png"), dpi=150)
    print("Saved fig4_lambda_vs_time.png")

    # ── Figure 5: Colony-size comparison (Spearman at last step) ─────────
    if len(all_results) > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_colony_size_comparison(all_results, labels, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "fig5_colony_comparison.png"),
                    dpi=150)
        print("Saved fig5_colony_comparison.png")

    # ── Print summary table ───────────────────────────────────────────────
    print("\n── Summary of protection ranges at last timepoint ──")
    #print(f"  Analytical steady-state prediction: λ = {LAMBDA_ANALYTICAL:.2f} µm")
    print(f"  {'Run':<20}  {'Step':>6}  {'λ half-max':>12}  {'λ Spearman':>12}")
    print("  " + "-" * 56)
    for results, label in zip(all_results, labels):
        r = results[-1]
        print(f"  {label:<20}  {r['step']:>6}  "
              f"{r['lambda_half_max']:>12.2f}  "
              f"{r['lambda_spearman']:>12.2f}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
