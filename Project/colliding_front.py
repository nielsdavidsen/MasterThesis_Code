import random
from CellModeller.Regulation.ModuleRegulator import ModuleRegulator
from CellModeller.Biophysics.BacterialModels.CLBacterium import CLBacterium
import numpy
import math
import os
import json

from CellModeller.Signalling.GridDiffusion import GridDiffusion #add
from CellModeller.Integration.CLCrankNicIntegrator import CLCrankNicIntegrator #add

# Two tall rectangles side by side: sensitive cells (cellType 0) on the left, resistant cells
# (cellType 1) on the right, separated by a small gap. Approximates a cutout of two touching annuli.
# Headless: run with Scripts/run_sim_server.py, model parameters are passed as env vars, e.g.
#   python Scripts/run_sim_server.py Project/colliding_front.py --param ANTIBIOTIC=5 --param W_S=40

gs = 4
max_cells = int(os.environ.get("MAX_CELLS", 10 * int(1e4)))  # default: 100,000

n_signals = 1
n_species = 1

lambda_s = 1.93
lambda_r = 1.93

ANTIBIOTIC = float(os.environ.get("ANTIBIOTIC", 10.0))            # initial antibiotic per grid cell
Diff_A = float(os.environ.get("ANTIBIOTIC_DIFF", 2000.0))         # default: 2000.0

# Initial colony
# The number of cells in each rectangle follows from DENSITY * HEIGHT * width
DENSITY = float(os.environ.get("DENSITY", 0.04))   # cells per um^2, same in both rectangles
HEIGHT = float(os.environ.get("HEIGHT", 400.0))    # height (y extent) of both rectangles
W_S = float(os.environ.get("W_S", 30.0))           # width of the sensitive rectangle (left)
W_R = float(os.environ.get("W_R", 30.0))           # width of the resistant rectangle (right)
GAP = float(os.environ.get("GAP", 6.0))            # empty gap between the rectangles, centred on x=0
SEED = os.environ.get("SEED")                      # optional random seed for reproducible layouts

# Initial cell geometry. The length is below the smallest targetVol (2.5), so the placed cells
# don't all divide in the first step (sim.addCell defaults to length 3.5)
CELL_LENGTH = 2.0
CELL_RADIUS = 0.5

# Diffusion grid sizing: the grid must hold the colony at max_cells (cells off the grid see no
# antibiotic), plus a buffer so the closed grid boundary doesn't distort the gradients at the colony edge
AREA_PER_CELL = 3.0   # um^2 per cell in a dense colony (measured ~2.5-2.9 in test runs)
GRID_BUFFER = 40.0    # um of free grid around the largest possible colony

# Constraints on the initial state
MAX_DENSITY = 0.08    # cells per um^2 (~22% area coverage), random placement jams well above this
MIN_SEP = 0.1         # extra clearance between cells when placing them
MAX_TRIES_PER_CELL = 1000


def random_dir():
            theta = random.uniform(0.0, 2*math.pi)
            return (math.cos(theta), math.sin(theta), 0)


def fail(msg):
    raise SystemExit('colliding_front: ' + msg)


def seg_dist(p1, q1, p2, q2):
    # Shortest distance between the 2D line segments p1-q1 and p2-q2 (cell axes)
    d1 = (q1[0]-p1[0], q1[1]-p1[1])
    d2 = (q2[0]-p2[0], q2[1]-p2[1])
    r = (p1[0]-p2[0], p1[1]-p2[1])
    a = d1[0]*d1[0] + d1[1]*d1[1]
    e = d2[0]*d2[0] + d2[1]*d2[1]
    f = d2[0]*r[0] + d2[1]*r[1]
    c = d1[0]*r[0] + d1[1]*r[1]
    b = d1[0]*d2[0] + d1[1]*d2[1]
    denom = a*e - b*b
    s = min(max((b*f - c*e) / denom, 0.0), 1.0) if denom > 1e-12 else 0.0
    t = (b*s + f) / e
    if t < 0.0:
        t = 0.0
        s = min(max(-c / a, 0.0), 1.0)
    elif t > 1.0:
        t = 1.0
        s = min(max((b - c) / a, 0.0), 1.0)
    dx = (p1[0] + d1[0]*s) - (p2[0] + d2[0]*t)
    dy = (p1[1] + d1[1]*s) - (p2[1] + d2[1]*t)
    return math.sqrt(dx*dx + dy*dy)


def cell_ends(pos, dir, length=CELL_LENGTH):
    h = 0.5 * length
    return ((pos[0] - dir[0]*h, pos[1] - dir[1]*h), (pos[0] + dir[0]*h, pos[1] + dir[1]*h))


class CellBins:
    # Spatial hash of cell axes, so each overlap test only looks at nearby cells
    def __init__(self, bin_size):
        self.bin_size = bin_size
        self.bins = {}

    def key(self, pos):
        return (int(math.floor(pos[0] / self.bin_size)), int(math.floor(pos[1] / self.bin_size)))

    def add(self, pos, ends):
        self.bins.setdefault(self.key(pos), []).append(ends)

    def min_dist(self, pos, ends):
        kx, ky = self.key(pos)
        dmin = float('inf')
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for other in self.bins.get((kx+i, ky+j), ()):
                    dmin = min(dmin, seg_dist(ends[0], ends[1], other[0], other[1]))
        return dmin


def place_cells(n, sample_pos, fits):
    # Random sequential placement: draw a position and orientation, keep the cell if it lies
    # inside the region and does not overlap any cell already placed
    bins = CellBins(CELL_LENGTH + 2*CELL_RADIUS + MIN_SEP)
    placed = []
    tries = 0
    while len(placed) < n:
        tries += 1
        if tries > MAX_TRIES_PER_CELL * n:
            fail('could only place %d of %d cells without overlaps - lower DENSITY'
                 % (len(placed), n))
        pos = sample_pos()
        dir = random_dir()
        ends = cell_ends(pos, dir)
        if not fits(ends):
            continue
        if bins.min_dist(pos, ends) < 2*CELL_RADIUS + MIN_SEP:
            continue
        bins.add(pos, ends)
        placed.append((pos, dir))
    return placed


def check_no_overlaps(sim):
    # Independent check on the cells as stored in the simulator: exit if any two cells overlap
    bins = CellBins(CELL_LENGTH + 2*CELL_RADIUS)
    for cs in sim.cellStates.values():
        ends = (tuple(cs.ends[0][:2]), tuple(cs.ends[1][:2]))
        d = bins.min_dist(cs.pos, ends)
        if d < 2*cs.radius:
            fail('initial state has overlapping cells (cell %d at (%.2f, %.2f), axis distance %.3f)'
                 % (cs.id, cs.pos[0], cs.pos[1], d))
        bins.add(cs.pos, ends)


def check_params():
    for name, val in (('DENSITY', DENSITY), ('HEIGHT', HEIGHT), ('W_S', W_S), ('W_R', W_R)):
        if val <= 0.0:
            fail('%s must be positive, got %g' % (name, val))
    if GAP < 0.0:
        fail('GAP must be non-negative')
    if min(HEIGHT, W_S, W_R) < CELL_LENGTH + 2*CELL_RADIUS:
        fail('HEIGHT, W_S and W_R must each be at least one cell length (%.1f)'
             % (CELL_LENGTH + 2*CELL_RADIUS))
    if DENSITY > MAX_DENSITY:
        fail('DENSITY %.4f cells/um^2 exceeds MAX_DENSITY %.4f' % (DENSITY, MAX_DENSITY))
    n_s = int(round(DENSITY * HEIGHT * W_S))
    n_r = int(round(DENSITY * HEIGHT * W_R))
    if n_s + n_r < 1:
        fail('DENSITY * HEIGHT * (W_S + W_R) gives no cells')
    if n_s + n_r >= max_cells:
        fail('initial cell count %d must be below MAX_CELLS (%d)' % (n_s + n_r, max_cells))
    return n_s, n_r

# Validate on import, i.e. before the Simulator creates an output folder for this run
N_S, N_R = check_params()

#Specify parameter for solving diffusion dynamics #Add
# Rectangular grid centred on the colony. Largest colony: every edge of the W x HEIGHT block advances
# by the same distance d until it holds max_cells, (W + 2d)(HEIGHT + 2d) = W*HEIGHT + max_cells*AREA_PER_CELL
W_TOTAL = W_S + GAP + W_R
D_GROWTH_MAX = (-(W_TOTAL + HEIGHT) + math.sqrt((W_TOTAL + HEIGHT)**2 + 4 * max_cells * AREA_PER_CELL)) / 4
nx_grid = 2 * int(math.ceil((W_TOTAL/2 + D_GROWTH_MAX + GRID_BUFFER) / gs))
ny_grid = 2 * int(math.ceil((HEIGHT/2 + D_GROWTH_MAX + GRID_BUFFER) / gs))
x_centre = (W_R - W_S) / 2  # the gap is centred on x=0, so the colony is off centre if W_S != W_R
grid_dim = (nx_grid, ny_grid, 8) # dimension of diffusion space, unit = number of grid
grid_size = (gs, gs, gs) # grid size
grid_orig = (x_centre - nx_grid//2 * gs, -ny_grid//2 * gs, -8) # where to place the diffusion space onto simulation space
grid_x = (grid_orig[0], grid_orig[0] + grid_dim[0]*gs)
grid_y = (grid_orig[1], grid_orig[1] + grid_dim[1]*gs)


def setup(sim):
    if SEED is not None:
        random.seed(int(SEED))

    # Set biophysics, signalling, and regulation models
    biophys = CLBacterium(sim, jitter_z=False, gamma = 50, max_cells=max_cells)
    sig = GridDiffusion(sim, n_signals, grid_dim, grid_size, grid_orig, D=[Diff_A], initLevels=[ANTIBIOTIC])
    integ = CLCrankNicIntegrator(sim, n_signals, n_species, max_cells, sig, boundcond='nearest')

    # use this file for reg too
    regul = ModuleRegulator(sim, sim.moduleName)
    # Only biophys and regulation
    sim.init(biophys, regul, sig, integ)

    # Each rectangle is x in [x0, x1], y in [-HEIGHT/2, HEIGHT/2], cell centres uniform over it
    def rectangle(x0, x1):
        def sample_pos():
            return (random.uniform(x0, x1), random.uniform(-HEIGHT/2, HEIGHT/2))
        # The whole cell (axis +- radius) must lie inside the rectangle
        def fits(ends):
            xs = (ends[0][0], ends[1][0])
            ys = (ends[0][1], ends[1][1])
            return (min(xs) >= x0 + CELL_RADIUS and max(xs) <= x1 - CELL_RADIUS and
                    min(ys) >= -HEIGHT/2 + CELL_RADIUS and max(ys) <= HEIGHT/2 - CELL_RADIUS)
        return sample_pos, fits

    # The rectangles are separated by GAP, so cells in different rectangles cannot overlap
    sensitive = place_cells(N_S, *rectangle(-GAP/2 - W_S, -GAP/2))
    resistant = place_cells(N_R, *rectangle(GAP/2, GAP/2 + W_R))

    for cellType, cells in ((0, sensitive), (1, resistant)):
        for pos, dir in cells:
            sim.addCell(
                cellType=cellType,
                pos=(pos[0], pos[1], 0),
                dir=dir,
                length=CELL_LENGTH,
                rad=CELL_RADIUS
            )

    check_no_overlaps(sim)

    # The parameters come from env vars and are not in the saved model source, so record them
    params = dict(N_CELLS=N_S + N_R, N_SENSITIVE=N_S, N_RESISTANT=N_R, DENSITY=DENSITY, HEIGHT=HEIGHT,
                  W_S=W_S, W_R=W_R, GAP=GAP, ANTIBIOTIC=ANTIBIOTIC,
                  ANTIBIOTIC_DIFF=Diff_A, MAX_CELLS=max_cells, SEED=SEED,
                  GRID_DIM=grid_dim, GRID_ORIG=grid_orig, GRID_SIZE=gs, D_GROWTH_MAX=D_GROWTH_MAX)
    print('colliding_front parameters: ' + json.dumps(params))
    if sim.saveOutput:
        with open(os.path.join(sim.outputDirPath, 'params.json'), 'w') as f:
            json.dump(params, f, indent=2)

    # No renderers: this version runs headless on the GPU server (see Scripts/run_sim_server.py)

    # Overridden by --pickle-steps when run through Scripts/run_sim_server.py
    sim.pickleSteps = 2

def init(cell):
    # Specify mean and distribution of initial cell size
    cell.targetVol = 2.5 + random.uniform(0.0,0.5)
    # Specify growth rate of cells
    GROWTH_RATES = {0: 0.1, 1: 1.0}
    cell.growthRate = GROWTH_RATES[cell.cellType]
    # Specify initial concentration of chemical species
    cell.species[:] = [0.0]*n_species
    # Specify initial concentration of signaling molecules
    cell.signals[:] = [0.0]*n_signals

cl_prefix = \
    '''

        const float D_A = 20.0f;
        const float k_A = 10.0f;

        float A_in = species[0];
        float A = signals[0];
        '''


def specRateCL():
    global cl_prefix
    # Remove diffusion in and out of cells

    return cl_prefix + '''

        if (cellType==0){
        // Antibiotic
        rates[0] = D_A*(A-A_in)*area/gridVolume;
        }

        else {
        // Antibiotic
        rates[0] = D_A*(A-A_in)*area/gridVolume - k_A*A_in;
        }
    '''

def sigRateCL():
    global cl_prefix
    return cl_prefix + '''

        if (cellType==0){
        // Antibiotic
        rates[0] = 0;
        }

        else {
        // Antibiotic
        rates[0] = - D_A*(A-A_in)*area/gridVolume;
        }
    '''


MIN_BRIGHTNESS = 0.4

def antibiotic_gr(A_in, IC_50=2.36):
     if A_in <= 4:
          return  1 / ( 1+ (A_in / IC_50)**3)
     elif 4 < A_in < 5:
          gs_4 = antibiotic_gr(4, IC_50)
          return gs_4 * (5-A_in)
     else:
          return 0.0


grid_warned = False

def update(cells):
    #Iterate through each cell and flag cells that reach target size for division
    global grid_warned


    # Antibiotic half-inhibition constant, so if A=IC_50, growth rate = 0.5*v_max
    IC_50 = 2.39

    for (id, cell) in cells.items():

        #Intracellular concentrations
        A_in = cell.species[0]


        if cell.cellType==0:
            antibiotic_factor = antibiotic_gr(A_in, IC_50=IC_50)
            cell.growthRate = (
                 lambda_s
                 * antibiotic_factor
            )
            cell.color = [0.1, 0.6, 0.1]

        else:
            cell.growthRate = lambda_r
            cell.color = [0.1, 0.1, 0.6]

        if cell.volume > cell.targetVol:
            cell.divideFlag = True

        # Cells off the diffusion grid see no antibiotic, warn once if that happens
        if not grid_warned and not (grid_x[0] <= cell.pos[0] < grid_x[1] and grid_y[0] <= cell.pos[1] < grid_y[1]):
            print('WARNING: cell %d at (%.1f, %.1f) is outside the diffusion grid x = [%g, %g], y = [%g, %g]'
                  % (id, cell.pos[0], cell.pos[1], grid_x[0], grid_x[1], grid_y[0], grid_y[1]), flush=True)
            grid_warned = True

def divide(parent, d1, d2):
    # Specify target cell size that triggers cell division
    d1.targetVol = 2.5 + random.uniform(0.0,0.5)
    d2.targetVol = 2.5 + random.uniform(0.0,0.5)
