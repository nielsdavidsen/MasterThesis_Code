import random
from CellModeller.Regulation.ModuleRegulator import ModuleRegulator
from CellModeller.Biophysics.BacterialModels.CLBacterium import CLBacterium
from CellModeller.GUI import Renderers
import numpy
import math

from CellModeller.Signalling.GridDiffusion import GridDiffusion #add
from CellModeller.Integration.CLCrankNicIntegrator import CLCrankNicIntegrator #add


gs = 4
max_cells = 1*int(1e5)

#Specify parameter for solving diffusion dynamics #Add
grid_dim = (160, 160, 8) # dimension of diffusion space, unit = number of grid
grid_size = (gs, gs, gs) # grid size
grid_orig = (-320, -320, -8) # where to place the diffusion space onto simulation space

n_signals = 1
n_species = 1

lambda_s = 1.93  
lambda_r = 1.93

ANTIBIOTIC = 10
Diff_A = 2000.0

def random_dir():
            theta = random.uniform(0.0, 2*math.pi)
            return (math.cos(theta), math.sin(theta), 0)


def add_ring(sim, center, R_ring, ring_width, spacing, ratio=0.01,
             theta_center=0.0, theta_span=2*math.pi):

    # Different radii across the thickness of the ring
    for r in numpy.arange(
        R_ring - ring_width/2,
        R_ring + ring_width/2,
        spacing
    ):

        # Number of cells needed around the full circumference, so that
        # cell spacing stays the same regardless of theta_span
        n_cells_full = max(1, int(2 * math.pi * r / spacing))
        n_cells_arc = max(1, int(round(n_cells_full * theta_span / (2*math.pi))))

        for i in range(n_cells_arc):

            frac = i / n_cells_arc if n_cells_arc > 1 else 0.5
            theta = theta_center - theta_span/2 + theta_span * frac

            x = center[0] + r * math.cos(theta)
            y = center[1] + r * math.sin(theta)
            rand = random.random()

            if rand < ratio:
                cellType = 1
            else:
                cellType = 0

            sim.addCell(
                cellType=cellType,
                pos=(x, y, 0),
                dir=random_dir()
            )

def setup(sim):
    # Set biophysics, signalling, and regulation models
    biophys = CLBacterium(sim, jitter_z=False, gamma = 50, max_cells=max_cells)
    sig = GridDiffusion(sim, n_signals, grid_dim, grid_size, grid_orig, D=[Diff_A], initLevels=[ANTIBIOTIC])
    integ = CLCrankNicIntegrator(sim, n_signals, n_species, max_cells, sig, boundcond='nearest')

    # use this file for reg too
    regul = ModuleRegulator(sim, sim.moduleName)
    # Only biophys and regulation
    sim.init(biophys, regul, sig, integ)

    
    
    R_ring = 30.0
    ring_width = 29.0
    spacing = 5.0

    center = (0.0, 0.0)
    ratio = 0.05

    # Only build the half of each ring facing the other colony (theta=0
    # points along +x, theta=pi along -x), which roughly halves the cell
    # count and speeds up the simulation.
    add_ring(sim, center, R_ring, ring_width, spacing, ratio=ratio,
             theta_center=0.0, theta_span=2*math.pi)

    # Add some objects to draw the models
    therenderer = Renderers.GLBacteriumRenderer(sim)
    sim.addRenderer(therenderer)
    sigrend = Renderers.GLGridRenderer(sig, integ, alpha = 0.3) # Add
    sim.addRenderer(sigrend) #Add

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

        const float D_A = 40.0f;
        const float k_A = 20.0f;

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
    

def update(cells):
    #Iterate through each cell and flag cells that reach target size for division


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

def divide(parent, d1, d2):
    # Specify target cell size that triggers cell division
    d1.targetVol = 2.5 + random.uniform(0.0,0.5)
    d2.targetVol = 2.5 + random.uniform(0.0,0.5)