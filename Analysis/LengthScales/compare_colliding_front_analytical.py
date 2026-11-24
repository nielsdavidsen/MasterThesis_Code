"""
compare_colliding_front_analytical.py
=====================================
Compares the antibiotic profile A(x, t) from the CellModeller model Project/colliding_front.py with
the analytical moving-boundary solution derived in MovingBoundaryAnalytical.ipynb.

What the script does
--------------------
1. Runs colliding_front.py headless (same way as Scripts/run_sim_server.py), in two phases:
   - pre-growth (--pregrow): the colonies grow with the antibiotic grid set to 0 until they are
     confluent. The colony is seeded sparsely (DENSITY <= 0.08), and while it fills in, the
     degradation per area (k_eff) keeps rising, whereas the analytical model has a constant k.
   - comparison (--t-end): at its start (t = 0) the antibiotic grid is reset to a uniform A0, the
     analytical initial condition, and pickles are written every --pickle-steps steps.
   colliding_front.py itself is not changed. Skip this with --run-dir to reuse a run.
2. For every pickle it reduces the 3D antibiotic grid to a 1D profile A(x) by averaging over y
   (only inside a central band of the colony, so the empty grid above/below the colony does not
   enter) and over the z layers of the grid.
3. It measures from the same pickles the parameters of the analytical model that the simulation
   does not set directly (see "Parameter mapping" below) and evaluates the analytical solution
   with them, so both use the same parameters.
4. It plots simulation (solid, with a +-1 standard deviation band over y) and analytical (dashed)
   profiles at the requested times, the residuals, and comparison statistics over time, and
   writes the statistics to comparison_statistics.csv (see "Statistics" below).

Statistics (per snapshot, evaluated at the analytical points inside the comparison region, i.e. up
to the right wall, or up to the resistant colony edge with --free-right-edge)
------------------------------------------------------------------------------------------------
RMSE, MAE        root mean square and mean absolute error of A_sim - A_an
bias             mean of A_sim - A_an (> 0: the simulation keeps more antibiotic)
max |err|        largest absolute error and the position where it occurs
NRMSE            RMSE / (A0 - min A_an), i.e. relative to the depth of the depletion
R^2              1 - SS_res / SS_tot of the analytical profile
within 1 SD      fraction of points where |A_sim - A_an| is below the y standard deviation of A_sim
A_I              antibiotic at the interface, simulation vs analytical
A_min            lowest antibiotic (resistant bulk), simulation vs analytical
lambda_1/2       distance from the interface into the sensitive side where the depletion
                 A0 - A has fallen to half its value at the interface (protection length)
deficit          integral of (A0 - A) dx over the comparison region, i.e. antibiotic removed per
                 unit y and z extent, simulation vs analytical

Parameter mapping (simulation -> analytical)
--------------------------------------------
A0   = ANTIBIOTIC             initial antibiotic, set identically in both.
D    = ANTIBIOTIC_DIFF / 6    GridDiffusion.transportRates computes laplace(u) * D / h^2 / 6, i.e. the
                              grid solves dA/dt = (ANTIBIOTIC_DIFF/6) del^2 A. --D is ANTIBIOTIC_DIFF.
L_S  = x_I(0) - left grid wall,  L_R = right grid wall - x_I(0).
       The grid walls are the only no-flux boundaries in the simulation (boundcond='nearest', so
       each wall lies half a grid spacing beyond the outermost node), which is the boundary
       condition of the analytical model at z = -L_S and z = L_R.
       By default the right wall is put exactly at the outer edge of the resistant block, so
       L_R = W_R (+ GAP/2) and the resistant cells fill the analytical resistant region. This is
       done without changing colliding_front.py: the script passes the model to the Simulator as
       source code (moduleStr) with a short grid override appended (GRID_OVERRIDE below), and that
       modified source is what gets saved with the run. Resistant cells later pushed beyond the
       wall are outside the domain. On the sensitive side the empty grid between the colony and
       the left wall is equivalent to the analytical sensitive region (no degradation in either).
       With --free-right-edge the grid keeps its usual buffer beyond the resistant colony, which
       has no degradation in the simulation but does in the analytical model; agreement is then
       only expected up to the resistant colony edge. Override with --L-S / --L-R.
k    = degradation rate, ONE constant value for the whole run (the same at every snapshot).
       colliding_front.py is used unchanged: every resistant cell removes k_A * A * volume / gridVolume
       from its grid nodes (volume = capsule volume pi r^2 (2r + l) that the OpenCL kernel receives,
       k_A read from the model source stored in the pickle). The y/z averaged concentration in a
       region of area S then decays at
           k_eff = k_A * sum(volume of resistant cells in S) / (S * depth_of_grid)
                   * A_cell_layer / A_depth_average.
       k_eff grows in time as the colony gets denser. The analytical k is its time average over the
       whole run, and measured_parameters.png shows k_eff(t) against that constant. --k-region
       chooses S: 'colony' (interface to resistant colony edge) or 'domain' (interface to right
       wall). Override with --k.
v    = interface velocity, x_I(t) = x_I(0) - v t, ONE value for the whole run: the slope of a
       linear fit of the measured x_I over all snapshots. The interface is found in strips along y
       as the midpoint between the rightmost sensitive and leftmost resistant cell. Override with --v.

Green's function leak (CellModeller artefact)
---------------------------------------------
CLCrankNicIntegrator computes its Crank-Nicolson Green's function with absorbing (zero) boundaries,
and on the shallow grid of colliding_front.py (8 layers in z) it reaches the z boundaries, so it
summed to slightly less than 1 (0.9926 for the default grid at dt = 0.05). Every step then removed
that fraction of antibiotic everywhere on the grid, independent of the cells. CLCrankNicIntegrator
now normalises the kernel and the pickles store its sum ('greensSum' = 1). For older runs without
'greensSum' the script rebuilds the Green's function exactly as CellModeller did and, unless
--no-leak-correction is given, divides the simulated profile at step n by sum(G)^n.

Usage
-----
Run the model and compare (uses the cellmodeller-thesis env, needs OpenCL and mpmath):
    python Analysis/LengthScales/compare_colliding_front_analytical.py --pregrow 1.5 --t-end 1

Re-analyse an existing run:
    python Analysis/LengthScales/compare_colliding_front_analytical.py --run-dir ~/cellmodeller/data/<run> --dt 0.05
"""

import os
import re
import sys
import glob
import json
import time
import pickle
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(REPO_ROOT, "Project", "colliding_front.py")

# Distinct categorical colours for the snapshot times, in a fixed order (colour-blind validated
# reference palette); a time keeps its colour across all figures
TIME_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SIM_COLOR, AN_COLOR, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"

# Appended to the colliding_front.py source (not the file) to put the right no-flux grid wall at the
# outer edge of the resistant block. Grid nodes sit at orig + i*gs and with boundcond='nearest' the
# wall lies half a grid spacing beyond the last node. The left wall stays where it was. grid_x is only
# used for the off-grid warning, so resistant cells pushed beyond the right wall do not trigger it.
GRID_OVERRIDE = '''

# ---- appended by Analysis/LengthScales/compare_colliding_front_analytical.py ----
# Right no-flux grid wall at the outer edge of the resistant block (resistant region = [x_I, wall])
X_WALL_RIGHT = GAP/2 + W_R
nx_grid = int(round((X_WALL_RIGHT - gs/2 - grid_orig[0]) / gs)) + 1
grid_dim = (nx_grid, ny_grid, 8)
grid_orig = (X_WALL_RIGHT - gs/2 - (nx_grid - 1) * gs, grid_orig[1], grid_orig[2])
grid_x = (grid_orig[0], float('inf'))
'''


# ── Command line ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("shared physical parameters (used by both model and analytical)")
    g.add_argument("--A0", type=float, default=10.0, help="initial antibiotic, ANTIBIOTIC (default 10)")
    g.add_argument("--D", type=float, default=2000.0, help="diffusion constant, ANTIBIOTIC_DIFF (default 2000)")

    g = p.add_argument_group("simulation setup (colliding_front.py env vars)")
    g.add_argument("--W-S", type=float, default=30.0, help="width of sensitive block (default 30)")
    g.add_argument("--W-R", type=float, default=100.0,
                   help="width of resistant block, equal to L_R unless --free-right-edge (default 100)")
    g.add_argument("--free-right-edge", action="store_true",
                   help="keep colliding_front.py's own grid instead of putting the right wall at the "
                        "resistant colony edge")
    g.add_argument("--height", type=float, default=200.0, help="height of the blocks (default 200)")
    g.add_argument("--density", type=float, default=0.08, help="initial cells per um^2 (default 0.08)")
    g.add_argument("--gap", type=float, default=0.0,
                   help="initial gap between blocks (default 0: the analytical model starts in contact)")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--max-cells", type=int, default=150000,
                   help="MAX_CELLS, also sets the grid size in colliding_front.py (default 150000)")
    g.add_argument("--pregrow", type=float, default=1.5,
                   help="time to grow the colonies without antibiotic before the comparison starts, so "
                        "that they are confluent and k_eff is nearly constant (default 1.5, 0 = none)")
    g.add_argument("--t-end", type=float, default=1.0,
                   help="duration of the comparison after the pre-growth (default 1)")
    g.add_argument("--dt", type=float, default=0.05, help="time step (default 0.05)")
    g.add_argument("--pickle-steps", type=int, default=2, help="steps between pickles (default 2)")
    g.add_argument("--out-root", default="~/cellmodeller", help="CellModeller output root (default ~/cellmodeller)")
    g.add_argument("--name", default=None, help="output folder name (default colliding_front_vs_analytical-<date>)")
    g.add_argument("--platform", type=int, default=0)
    g.add_argument("--device", type=int, default=0)
    g.add_argument("--run-dir", default=None, help="analyse this existing run instead of running the model")

    g = p.add_argument_group("analysis")
    g.add_argument("--times", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75, 1.0],
                   help="times after the start of the comparison, nearest pickle is used "
                        "(default 0.1 0.25 0.5 0.75 1)")
    g.add_argument("--y-band", type=float, default=0.5,
                   help="average over |y - y_centre| < y_band * HEIGHT/2 (default 0.5)")
    g.add_argument("--strip", type=float, default=4.0, help="strip height for interface detection (default 4)")
    g.add_argument("--k-region", choices=["colony", "domain"], default="colony",
                   help="region over which k_eff is measured, see docstring (default colony)")
    g.add_argument("--k", type=float, default=None, help="override k for the analytical model")
    g.add_argument("--v", type=float, default=None, help="override v for the analytical model")
    g.add_argument("--L-S", type=float, default=None, help="override L_S for the analytical model")
    g.add_argument("--L-R", type=float, default=None, help="override L_R for the analytical model")
    g.add_argument("--no-leak-correction", action="store_true",
                   help="do not correct the simulated profiles for the Green's function leak")
    g.add_argument("--n-space", type=int, default=81, help="analytical points per region (default 81)")
    g.add_argument("--dps", type=int, default=30, help="mpmath precision (default 30)")
    return p.parse_args()


# ── 1. Run the model ─────────────────────────────────────────────────────────

def run_model(args):
    """Run colliding_front.py headless until t_end, mirroring Scripts/run_sim_server.py."""
    env = dict(ANTIBIOTIC=args.A0, ANTIBIOTIC_DIFF=args.D, W_S=args.W_S, W_R=args.W_R,
               HEIGHT=args.height, DENSITY=args.density, GAP=args.gap, SEED=args.seed,
               MAX_CELLS=args.max_cells)
    for key, val in env.items():
        os.environ[key] = str(val)

    out_root = os.path.abspath(os.path.expanduser(args.out_root))
    os.makedirs(os.path.join(out_root, "data"), exist_ok=True)
    os.environ["CMPATH"] = out_root
    name = args.name or "colliding_front_vs_analytical-" + time.strftime("%y-%m-%d-%H-%M")

    sys.path.insert(0, os.path.dirname(MODEL_PATH))
    from CellModeller.Simulator import Simulator
    with open(MODEL_PATH, encoding="utf-8") as f:
        model_src = f.read()
    if not args.free_right_edge:
        model_src += GRID_OVERRIDE
    sim = Simulator("colliding_front", args.dt, outputDirName=name, moduleStr=model_src,
                    clPlatformNum=args.platform, clDeviceNum=args.device, saveOutput=True)
    # Sim.step() writes step-N.pickle *after* step N+1, and never saves the starting state. Switch that
    # off and write the pickles here instead, at the start of the comparison and every pickle_steps
    sim.saveOutput = False
    cell_limit = args.max_cells - max(256, int(0.05 * args.max_cells))
    start = time.time()

    def advance(n, pickle_from=None):
        """Step n times; write a pickle every pickle_steps steps counted from pickle_from."""
        for _ in range(n):
            if len(sim.cellStates) >= cell_limit:
                print("Stopping early at step %d: reached %d cells (MAX_CELLS %d)"
                      % (sim.stepNum, len(sim.cellStates), args.max_cells))
                return False
            sim.step()
            if pickle_from is not None and (sim.stepNum - pickle_from) % args.pickle_steps == 0:
                sim.writePickle()
            if sim.stepNum % 20 == 0:
                print("step %5d   cells %7d   elapsed %7.1f s"
                      % (sim.stepNum, len(sim.cellStates), time.time() - start), flush=True)
        return True

    # Pre-growth: grow the colonies without antibiotic until they are confluent, so that the
    # degradation per area (k_eff) is nearly constant during the comparison, as in the analytical model
    n_pre = int(round(args.pregrow / args.dt))
    if n_pre > 0:
        sim.integ.signalLevel[:] = 0.0
        print("Pre-growth without antibiotic for %d steps (%g time units)" % (n_pre, args.pregrow))
        advance(n_pre)

    # Start of the comparison (t = 0): uniform antibiotic A0 everywhere, the analytical initial condition
    sim.integ.signalLevel[:] = args.A0
    sim.integ.specLevel[:] = 0.0
    t0_step = sim.stepNum
    sim.writePickle()
    with open(os.path.join(sim.outputDirPath, "comparison_run.json"), "w") as f:
        json.dump(dict(dt=args.dt, t_end=args.t_end, pregrow=args.pregrow, t0_step=t0_step,
                       pickle_steps=args.pickle_steps, env=env,
                       right_wall_at_resistant_edge=not args.free_right_edge,
                       pickle_time="t = (stepNum - t0_step) * dt"), f, indent=2)

    n_steps = int(round(args.t_end / args.dt))
    print("Comparison: %d steps (t_end = %g) from %d cells, output: %s"
          % (n_steps, args.t_end, len(sim.cellStates), sim.outputDirPath))
    advance(n_steps, pickle_from=t0_step)
    if (sim.stepNum - t0_step) % args.pickle_steps != 0:
        sim.writePickle()
    return sim.outputDirPath


# ── 2. Reduce pickles ────────────────────────────────────────────────────────

def read_timing(run_dir, dt_arg):
    """Return dt and the step offset such that a pickle step-N.pickle holds t = (N + offset) * dt."""
    path = os.path.join(run_dir, "comparison_run.json")
    if os.path.exists(path):
        with open(path) as f:
            info = json.load(f)                # written by this script
        return info["dt"], -info.get("t0_step", 0)
    return dt_arg, 1                          # written by Simulator.step(), e.g. via run_sim_server.py


def read_k_A(module_str):
    m = re.search(r"const\s+float\s+k_A\s*=\s*([0-9.eE+-]+)f?", module_str)
    if not m:
        raise SystemExit("Could not find k_A in the model source stored in the pickle")
    return float(m.group(1))


def analyse_pickle(path, params, args):
    """Return the 1D profile and the quantities needed for the analytical parameters."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    n_sig, nx, ny, nz = data["sigGridDim"]
    gs = float(data["sigGridSize"][0])
    orig = np.asarray(data["sigGridOrig"], dtype=float)
    grid = np.asarray(data["sigGrid"]).reshape(data["sigGridDim"])[0]   # (nx, ny, nz)

    # Grid nodes sit at orig + i*gs (see GridDiffusion.trilinearWeights)
    x_nodes = orig[0] + gs * np.arange(nx)
    y_nodes = orig[1] + gs * np.arange(ny)
    y_c = 0.0                                   # the blocks are centred on y = 0
    half_band = args.y_band * params["HEIGHT"] / 2
    in_band = np.abs(y_nodes - y_c) <= half_band
    A_rows = grid[:, in_band, :].mean(axis=2)                   # depth average, one row per y node
    A_depth = A_rows.mean(axis=1)                               # average over y band and z
    A_std = A_rows.std(axis=1)                                  # spread over y of the depth average
    iz_cells = int(round((0.0 - orig[2]) / gs))                 # z layer that holds the cells
    A_layer = grid[:, in_band, iz_cells].mean(axis=1)

    cells = data["cellStates"]
    pos = np.array([c.pos[:2] for c in cells.values()], dtype=float)
    ctype = np.array([c.cellType for c in cells.values()], dtype=int)
    # The volume passed to sigRateCL is cell_vols_dev = pi r^2 (2r + l) (CLBacterium.calc_cell_vol),
    # not cellState.volume (which CLBacterium sets to the length)
    vol = np.array([np.pi * c.radius**2 * (2 * c.radius + c.length) for c in cells.values()], dtype=float)
    band = np.abs(pos[:, 1] - y_c) <= half_band
    sens = band & (ctype == 0)
    res = band & (ctype == 1)

    # Interface: per y strip midpoint between rightmost sensitive and leftmost resistant cell
    edges = np.arange(-half_band, half_band + args.strip, args.strip)
    mids = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = sens & (pos[:, 1] >= lo) & (pos[:, 1] < hi)
        r = res & (pos[:, 1] >= lo) & (pos[:, 1] < hi)
        if s.any() and r.any():
            mids.append(0.5 * (pos[s, 0].max() + pos[r, 0].min()))
    x_I = float(np.median(mids)) if mids else np.nan

    # Effective degradation rate of the y/z averaged concentration
    # No-flux walls lie half a grid spacing beyond the outermost nodes (boundcond='nearest')
    x_wall_left, x_wall_right = x_nodes[0] - gs / 2, x_nodes[-1] + gs / 2
    # Fraction of each cell's trilinear weight (in x) that lands on grid nodes; CellModeller drops
    # the weight on nodes outside the grid
    w_grid = np.clip(1 - (pos[:, 0] - x_nodes[-1]) / gs, 0, 1) * np.clip(1 + (pos[:, 0] - x_nodes[0]) / gs, 0, 1)
    x_res_edge = np.percentile(pos[res, 0], 99.5) if res.any() else np.nan
    x_right = min(x_res_edge, x_wall_right) if args.k_region == "colony" else x_wall_right
    area = (x_right - x_I) * 2 * half_band
    depth = nz * gs
    region = (x_nodes >= x_I) & (x_nodes <= x_right)
    k_eff = params["k_A"] * (vol * w_grid)[res].sum() / (area * depth) if area > 0 else np.nan
    # Cells degrade the antibiotic of their own z layer, which is lower than the depth average.
    # The depth averaged equation then has k_eff * A_layer instead of k_eff * A_depth, so scale by
    # the ratio inside the region (exact reduction to 1D for the depth average)
    if region.any():
        k_eff *= A_layer[region].mean() / A_depth[region].mean()

    return dict(step=data["stepNum"], x=x_nodes, A=A_depth, A_std=A_std, A_layer=A_layer, x_I=x_I, k_eff=k_eff,
                x_res_edge=x_res_edge, x_wall=(x_wall_left, x_wall_right), n_cells=len(cells))


def greens_sum(grid_dim, gs, D, dt):
    """Sum of the Green's function exactly as CLCrankNicIntegrator.computeGreensFunc builds it."""
    from scipy.sparse.linalg import LinearOperator, gmres
    from scipy.ndimage import laplace
    dims = tuple(int(d) for d in grid_dim)
    n = int(np.prod(dims))
    rate = D / gs**2 / 6.0

    def cn_operator(v):
        # (I - dt/2 T) v, with T evaluated in 'greens' mode: boundcond 'constant', boundary value 0
        return v - 0.5 * dt * laplace(v.reshape(dims), None, mode="constant", cval=0.0).ravel() * rate

    L = LinearOperator((n, n), matvec=cn_operator, dtype=np.float32)
    rhs = np.zeros(dims, dtype=np.float32)
    rhs[tuple(d // 2 for d in dims)] = 1.0          # interpAddToGrid at the grid centre (node aligned)
    G, _ = gmres(L, rhs.ravel())
    return float(G.sum())


# ── 3. Analytical solution (as derived in MovingBoundaryAnalytical.ipynb) ─────

class Analytical:
    """Moving-boundary solution, sensitive z in [-L_S, 0], resistant z in [0, L_R], x_I = -v t."""

    def __init__(self, A0, D, k, L_S, L_R, v):
        self.A0, self.D, self.k, self.L_S, self.L_R, self.v = A0, D, k, L_S, L_R, v

    def _components(self, s):
        A0, D, k, L_S, L_R, v = self.A0, self.D, self.k, self.L_S, self.L_R, self.v
        a = v / (2 * D)
        mu_s = mp.sqrt(v**2 + 4 * D * s) / (2 * D)
        mu_r = mp.sqrt(v**2 + 4 * D * (s + k)) / (2 * D)
        hs = mp.exp(a * L_S) * (mp.cosh(mu_s * L_S) - (a / mu_s) * mp.sinh(mu_s * L_S))
        hs_prime = mp.exp(a * L_S) * ((s / (D * mu_s)) * mp.sinh(mu_s * L_S))
        hr = mp.exp(-a * L_R) * (mp.cosh(mu_r * L_R) + (a / mu_r) * mp.sinh(mu_r * L_R))
        hr_prime = mp.exp(-a * L_R) * (((s + k) / (D * mu_r)) * mp.sinh(mu_r * L_R))
        g_s = hs_prime / hs
        g_r = -hr_prime / hr
        A_I_hat = A0 * (g_s / s - g_r / (s + k)) / (g_s - g_r)
        return a, mu_s, mu_r, A_I_hat

    def sensitive_hat(self, z, s):
        a, mu_s, _, A_I_hat = self._components(s)
        H = lambda y: mp.exp(a * y) * (mp.cosh(mu_s * y) - (a / mu_s) * mp.sinh(mu_s * y))
        return self.A0 / s + (A_I_hat - self.A0 / s) * H(z + self.L_S) / H(self.L_S)

    def resistant_hat(self, z, s):
        a, _, mu_r, A_I_hat = self._components(s)
        k = self.k
        H = lambda y: mp.exp(-a * y) * (mp.cosh(mu_r * y) + (a / mu_r) * mp.sinh(mu_r * y))
        return self.A0 / (s + k) + (A_I_hat - self.A0 / (s + k)) * H(self.L_R - z) / H(self.L_R)

    @staticmethod
    def _invert(F, t):
        return float(mp.re(mp.invertlaplace(F, t, method="dehoog")))

    def profile(self, t, n):
        """Return x relative to the initial interface (lab frame, x = z - v t) and A."""
        z_s = np.linspace(-self.L_S, 0, n)
        z_r = np.linspace(0, self.L_R, n)
        A_s = [self._invert(lambda s, z=z: self.sensitive_hat(z, s), t) for z in z_s]
        A_r = [self._invert(lambda s, z=z: self.resistant_hat(z, s), t) for z in z_r[1:]]
        z = np.concatenate([z_s, z_r[1:]])
        return z - self.v * t, np.array(A_s + A_r)


# ── 4. Comparison statistics ─────────────────────────────────────────────────

def half_depletion_length(x, A, x_I, A0):
    """Distance from x_I into the sensitive side (x < x_I) where A0 - A falls to half its value at x_I."""
    dep_I = A0 - np.interp(x_I, x, A)
    if dep_I <= 0:
        return np.nan
    xs = np.concatenate([[x_I], x[x < x_I][::-1]])              # walk left from the interface
    dep = A0 - np.interp(xs, x, A)
    below = np.nonzero(dep <= 0.5 * dep_I)[0]
    if len(below) == 0:
        return np.nan
    j = below[0]
    # linear interpolation between xs[j-1] and xs[j]
    f = (dep[j - 1] - 0.5 * dep_I) / (dep[j - 1] - dep[j])
    return x_I - (xs[j - 1] + f * (xs[j] - xs[j - 1]))


def comparison_stats(r, A0, x_right):
    """Statistics of simulation vs analytical at the analytical points with x <= x_right."""
    m = r["x_an"] <= x_right
    x = r["x_an"][m]
    A_an = r["A_an"][m]
    A_sim = np.interp(x, r["x_sim"], r["A_sim"])
    sd_sim = np.interp(x, r["x_sim"], r["A_std"])
    err = A_sim - A_an
    i_max = int(np.argmax(np.abs(err)))
    ss_tot = np.sum((A_an - A_an.mean()) ** 2)
    x_I_an = -r["v"] * r["t"]                                   # analytical interface, lab frame
    return dict(
        t=r["t"], k=r["k"], v=r["v"],
        rmse=float(np.sqrt(np.mean(err ** 2))),
        mae=float(np.mean(np.abs(err))),
        bias=float(np.mean(err)),
        max_abs_err=float(np.abs(err[i_max])),
        x_max_abs_err=float(x[i_max]),
        nrmse=float(np.sqrt(np.mean(err ** 2)) / (A0 - A_an.min())) if A_an.min() < A0 else np.nan,
        r2=float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else np.nan,
        frac_within_1sd=float(np.mean(np.abs(err) <= sd_sim)),
        A_I_sim=float(np.interp(r["x_I"], r["x_sim"], r["A_sim"])),
        A_I_an=float(np.interp(x_I_an, r["x_an"], r["A_an"])),
        A_min_sim=float(A_sim.min()),
        A_min_an=float(A_an.min()),
        lambda_half_sim=float(half_depletion_length(r["x_sim"], r["A_sim"], r["x_I"], A0)),
        lambda_half_an=float(half_depletion_length(r["x_an"], r["A_an"], x_I_an, A0)),
        deficit_sim=float(np.trapezoid(A0 - A_sim, x)),
        deficit_an=float(np.trapezoid(A0 - A_an, x)),
    )


STAT_COLUMNS = [
    ("t", "t", "%.2f"), ("k", "k", "%.4f"), ("v", "v", "%.3f"),
    ("rmse", "RMSE", "%.4f"), ("mae", "MAE", "%.4f"), ("bias", "bias", "%+.4f"),
    ("max_abs_err", "max|err|", "%.4f"), ("x_max_abs_err", "at x", "%.1f"),
    ("nrmse", "NRMSE", "%.3f"), ("r2", "R^2", "%.4f"), ("frac_within_1sd", "in 1SD", "%.2f"),
    ("A_I_sim", "A_I sim", "%.3f"), ("A_I_an", "A_I an", "%.3f"),
    ("A_min_sim", "A_min sim", "%.3f"), ("A_min_an", "A_min an", "%.3f"),
    ("lambda_half_sim", "l1/2 sim", "%.1f"), ("lambda_half_an", "l1/2 an", "%.1f"),
    ("deficit_sim", "deficit sim", "%.1f"), ("deficit_an", "deficit an", "%.1f"),
]


def style_axes(ax):
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ── 5. Main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    mp.mp.dps = args.dps

    run_dir = args.run_dir and os.path.abspath(os.path.expanduser(args.run_dir))
    if run_dir is None:
        run_dir = run_model(args)

    with open(os.path.join(run_dir, "params.json")) as f:
        params = json.load(f)
    dt, step_offset = read_timing(run_dir, args.dt)
    run_info_path = os.path.join(run_dir, "comparison_run.json")
    run_info = json.load(open(run_info_path)) if os.path.exists(run_info_path) else {}
    wall_fit = bool(run_info.get("right_wall_at_resistant_edge", False))
    pickles = sorted(glob.glob(os.path.join(run_dir, "step-*.pickle")))
    if not pickles:
        raise SystemExit("No pickles in %s" % run_dir)
    with open(pickles[0], "rb") as f:
        first = pickle.load(f)
    params["k_A"] = read_k_A(first["moduleStr"])
    # Runs made after the leak fix in CLCrankNicIntegrator store the kernel sum (1.0) in the pickle
    stored_g_sum = float(np.ravel(first["greensSum"])[0]) if "greensSum" in first else None

    # The shared parameters must agree between the run and the analytical model
    A0, D_sim = params["ANTIBIOTIC"], params["ANTIBIOTIC_DIFF"]
    if args.run_dir and (A0 != args.A0 or D_sim != args.D):
        print("Note: using A0 = %g and ANTIBIOTIC_DIFF = %g from the run's params.json" % (A0, D_sim))
    D = D_sim / 6.0     # diffusion constant the grid actually solves, see docstring

    print("Reducing %d pickles ..." % len(pickles))
    snaps = [analyse_pickle(p, params, args) for p in pickles]

    if stored_g_sum is not None:
        g_sum = stored_g_sum
        print("Green's function sum stored in the run = %.6f" % g_sum)
    else:
        g_sum = greens_sum(params["GRID_DIM"], params["GRID_SIZE"], D_sim, dt)
        print("Run predates the leak fix, rebuilt Green's function sum = %.6f: the solver removed "
              "%.3f%% of the antibiotic per step" % (g_sum, 100 * (1 - g_sum)))
    leak_corrected = not args.no_leak_correction and abs(g_sum - 1.0) > 1e-6
    if leak_corrected:
        for s in snaps:
            n_diff_steps = s["step"] + step_offset           # number of integration steps done
            for key in ("A", "A_std", "A_layer"):
                s[key] = s[key] / g_sum**n_diff_steps
    t_all = np.array([(s["step"] + step_offset) * dt for s in snaps])
    x_I_all = np.array([s["x_I"] for s in snaps])
    k_all = np.array([s["k_eff"] for s in snaps])
    x_I0 = x_I_all[np.isfinite(x_I_all)][0]
    x_wall_left, x_wall_right = snaps[0]["x_wall"]

    L_S = args.L_S if args.L_S is not None else x_I0 - x_wall_left
    L_R = args.L_R if args.L_R is not None else x_wall_right - x_I0

    out_dir = os.path.join(run_dir, "analytical_comparison")
    os.makedirs(out_dir, exist_ok=True)

    # One k and one v for the whole run, used at every snapshot
    ok = np.isfinite(x_I_all) & np.isfinite(k_all)
    if args.v is not None:
        v, v_source = args.v, "--v"
    elif ok.sum() >= 2:
        v, v_source = -np.polyfit(t_all[ok], x_I_all[ok], 1)[0], "linear fit of x_I over the whole run"
    else:
        v, v_source = 0.0, "too few snapshots, set to 0"
    if args.k is not None:
        k, k_source = args.k, "--k"
    else:
        k = np.trapezoid(k_all[ok], t_all[ok]) / (t_all[ok][-1] - t_all[ok][0])
        k_source = "time average of the measured k_eff over the comparison"
    print("Analytical parameters (same for every snapshot): k = %.4g (%s), v = %.4g (%s)"
          % (k, k_source, v, v_source))
    k_lo, k_hi = np.nanmin(k_all[ok]), np.nanmax(k_all[ok])
    print("Measured k_eff of the simulation ranges from %.4g to %.4g during the comparison" % (k_lo, k_hi))
    if (k_hi - k_lo) > 0.1 * k:
        print("WARNING: k_eff varies by %.0f%% of k during the comparison, so the constant-k analytical model "
              "is only approximate. A longer --pregrow makes the colony more confluent first."
              % (100 * (k_hi - k_lo) / k))

    results = []
    used = []
    for t_req in args.times:
        i = int(np.argmin(np.abs(t_all - t_req)))
        if i in used:
            continue
        used.append(i)
        t = t_all[i]
        if abs(t - t_req) > 1e-9:
            print("t = %g not saved, using nearest pickle at t = %g" % (t_req, t))
        if t <= 0:
            print("Skipping t = %g (analytical solution needs t > 0)" % t_req)
            continue
        print("t = %.3f: A0 = %g, D = %g, k = %.4g, v = %.4g, L_S = %.1f, L_R = %.1f  (interface at %.2f, "
              "resistant edge at %.1f)" % (t, A0, D, k, v, L_S, L_R, snaps[i]["x_I"] - x_I0,
                                           snaps[i]["x_res_edge"] - x_I0))
        model = Analytical(A0, D, k, L_S, L_R, v)
        x_an, A_an = model.profile(t, args.n_space)
        x_sim = snaps[i]["x"] - x_I0
        results.append(dict(t=t, k=k, v=v, x_an=x_an, A_an=A_an, x_sim=x_sim, A_sim=snaps[i]["A"],
                            A_std=snaps[i]["A_std"], A_layer=snaps[i]["A_layer"],
                            x_I=snaps[i]["x_I"] - x_I0, x_res_edge=snaps[i]["x_res_edge"] - x_I0))

    if not results:
        raise SystemExit("Nothing to compare")

    # Comparison region: the whole domain if the resistant colony fills it up to the wall, otherwise
    # only up to the resistant colony edge (beyond it the two models differ by construction)
    stats = []
    for r in results:
        r["x_right"] = (x_wall_right - x_I0) if wall_fit else min(r["x_res_edge"], x_wall_right - x_I0)
        stats.append(comparison_stats(r, A0, r["x_right"]))
    colors = [TIME_COLORS[j % len(TIME_COLORS)] for j in range(len(results))]
    times = np.array([s["t"] for s in stats])

    # ── Profiles: simulation +- 1 SD over y (band), analytical dashed
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for r, c in zip(results, colors):
        ax.fill_between(r["x_sim"], r["A_sim"] - r["A_std"], r["A_sim"] + r["A_std"], color=c, alpha=0.18, lw=0)
        ax.plot(r["x_sim"], r["A_sim"], color=c, lw=2, label="t = %.2f" % r["t"])
        ax.plot(r["x_an"], r["A_an"], color=c, lw=2, ls="--")
        ax.axvline(r["x_I"], color=c, lw=1, ls=":")
        if not wall_fit:
            ax.axvline(r["x_res_edge"], color=c, lw=1, ls="-.")
    ax.plot([], [], color=MUTED, lw=2, label="CellModeller, mean over y and z%s"
            % (" (leak corrected)" if leak_corrected else ""))
    ax.fill_between([], [], [], color=MUTED, alpha=0.25, lw=0, label="CellModeller, +-1 SD over y")
    ax.plot([], [], color=MUTED, lw=2, ls="--", label="analytical")
    ax.plot([], [], color=MUTED, lw=1, ls=":", label="interface")
    if not wall_fit:
        ax.plot([], [], color=MUTED, lw=1, ls="-.", label="outer edge of resistant colony")
    ax.axvline(x_wall_right - x_I0, color=INK, lw=1.5, label="right grid wall (z = L_R)")
    ax.set_xlabel("x - x_I(0)  [um]")
    ax.set_ylabel("antibiotic A")
    ax.set_title("Antibiotic profile, A0 = %g, D = %g (ANTIBIOTIC_DIFF/6), k = %.3g, v = %.3g, L_S = %.0f, L_R = %.0f"
                 % (A0, D, k, v, L_S, L_R), fontsize=10)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "profiles.png"), dpi=150)

    # Zoomed version around the colonies
    margin = 3 * np.sqrt(D * results[-1]["t"])      # three diffusion lengths
    lo = min(r["x_I"] for r in results) - margin
    hi = min(max(r["x_res_edge"] for r in results) + margin, x_wall_right - x_I0 + 10)
    ax.set_xlim(lo, hi)
    fig.savefig(os.path.join(out_dir, "profiles_zoom.png"), dpi=150)
    plt.close(fig)

    # ── Residuals A_sim - A_an with the simulation's +- 1 SD band around zero
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for r, c in zip(results, colors):
        m = r["x_an"] <= r["x_right"]
        x = r["x_an"][m]
        err = np.interp(x, r["x_sim"], r["A_sim"]) - r["A_an"][m]
        sd = np.interp(x, r["x_sim"], r["A_std"])
        ax.fill_between(x, -sd, sd, color=c, alpha=0.15, lw=0)
        ax.plot(x, err, color=c, lw=2, label="t = %.2f" % r["t"])
    ax.axhline(0, color=INK, lw=1)
    ax.fill_between([], [], [], color=MUTED, alpha=0.25, lw=0, label="+-1 SD of simulation over y")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("x - x_I(0)  [um]")
    ax.set_ylabel("A_sim - A_analytical")
    ax.set_title("Residuals (simulation - analytical)", fontsize=10)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "residuals.png"), dpi=150)
    plt.close(fig)

    # ── Statistics over time: errors, and simulation vs analytical for derived quantities.
    # One quantity per panel; simulation and analytical share that panel's single axis.
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    ax = axes[0, 0]
    for key, label, c in (("rmse", "RMSE", "#2a78d6"), ("mae", "MAE", "#1baf7a"),
                          ("max_abs_err", "max |err|", "#eb6834")):
        ax.plot(times, [s[key] for s in stats], color=c, lw=2, marker="o", ms=5, label=label)
    ax.set_ylabel("error in A")
    ax.set_title("Error (simulation - analytical)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[0, 1]
    ax.plot(times, [s["r2"] for s in stats], color="#2a78d6", lw=2, marker="o", ms=5, label="R^2")
    ax.plot(times, [s["frac_within_1sd"] for s in stats], color="#1baf7a", lw=2, marker="o", ms=5,
            label="fraction within 1 SD")
    finite_r2 = [s["r2"] for s in stats if np.isfinite(s["r2"])]
    ax.set_ylim(min([0.0] + [r2 - 0.05 for r2 in finite_r2]), 1.05)
    ax.set_title("Goodness of fit", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    for ax, (k_sim, k_an, title, ylabel) in zip(
            [axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]],
            [("A_I_sim", "A_I_an", "Antibiotic at the interface", "A_I"),
             ("A_min_sim", "A_min_an", "Lowest antibiotic (resistant bulk)", "A_min"),
             ("lambda_half_sim", "lambda_half_an", "Half-depletion length into sensitive side", "lambda_1/2 [um]"),
             ("deficit_sim", "deficit_an", "Antibiotic deficit, integral of (A0 - A) dx", "deficit [A um]")]):
        ax.plot(times, [s[k_sim] for s in stats], color=SIM_COLOR, lw=2, marker="o", ms=5, label="CellModeller")
        ax.plot(times, [s[k_an] for s in stats], color=AN_COLOR, lw=2, ls="--", marker="s", ms=5, label="analytical")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8)
    for ax in axes.ravel():
        ax.set_xlabel("time")
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "statistics.png"), dpi=150)
    plt.close(fig)

    # ── Measured parameters vs time, one panel each (no shared y axis)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(t_all, x_I_all - x_I0, color=SIM_COLOR, lw=2, marker="o", ms=3, label="CellModeller")
    axes[0].plot(t_all, -v * t_all, color=AN_COLOR, lw=2, ls="--", label="analytical, v = %.3g" % v)
    axes[0].set_ylabel("interface position x_I - x_I(0) [um]")
    axes[1].plot(t_all, k_all, color=SIM_COLOR, lw=2, marker="o", ms=3, label="CellModeller, measured")
    axes[1].axhline(k, color=AN_COLOR, lw=2, ls="--", label="analytical, k = %.3g" % k)
    axes[1].set_ylabel("effective degradation rate k [1/time]")
    axes[1].set_ylim(bottom=0)
    for a in axes:
        a.set_xlabel("time")
        style_axes(a)
        a.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "measured_parameters.png"), dpi=150)
    plt.close(fig)

    # ── Data and statistics files
    np.savez(os.path.join(out_dir, "comparison.npz"), t=t_all, x_I=x_I_all - x_I0, k_eff=k_all,
             **{"%s_%d" % (key, j): r[key] for j, r in enumerate(results)
                for key in ("x_sim", "A_sim", "A_std", "A_layer", "x_an", "A_an")})
    with open(os.path.join(out_dir, "comparison_statistics.csv"), "w") as f:
        f.write(",".join(key for key, _, _ in STAT_COLUMNS) + "\n")
        for s in stats:
            f.write(",".join(repr(s[key]) for key, _, _ in STAT_COLUMNS) + "\n")
    with open(os.path.join(out_dir, "analytical_parameters.json"), "w") as f:
        json.dump(dict(A0=A0, D=D, ANTIBIOTIC_DIFF=D_sim, greens_sum=g_sum, leak_corrected=leak_corrected,
                       right_wall_at_resistant_edge=wall_fit, L_S=L_S, L_R=L_R, k_A=params["k_A"],
                       k_region=args.k_region, y_band=args.y_band, dt=dt, snapshots=stats), f, indent=2)

    # ── Statistics table in the terminal
    print("\nComparison statistics (region: x <= %s)" % ("right wall" if wall_fit else "resistant colony edge"))
    widths = [max(len(head), 7) for _, head, _ in STAT_COLUMNS]
    print("  ".join(head.rjust(w) for (_, head, _), w in zip(STAT_COLUMNS, widths)))
    for s in stats:
        print("  ".join((fmt % s[key]).rjust(w) for (key, _, fmt), w in zip(STAT_COLUMNS, widths)))
    print("\nFigures and statistics written to %s" % out_dir)


if __name__ == "__main__":
    main()
