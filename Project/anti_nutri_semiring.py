import random
from CellModeller.Regulation.ModuleRegulator import ModuleRegulator
from CellModeller.Biophysics.BacterialModels.CLBacterium import CLBacterium
from CellModeller.GUI import Renderers
import numpy
import math

from CellModeller.Signalling.GridDiffusion import GridDiffusion #add
from CellModeller.Integration.CLCrankNicIntegrator import CLCrankNicIntegrator #add


max_cells = 20*int(1e4)

#Specify parameter for solving diffusion dynamics #Add
grid_dim = (160, 160, 8) # dimension of diffusion space, unit = number of grid
grid_size = (2, 2, 2) # grid size
grid_orig = (-160, -160, -8) # where to place the diffusion space onto simulation space

n_signals = 2
n_species = 2


ANTIBIOTIC = 5.0
NUTRIENTS = 5.0

Diff_A = 10.0
Diff_N = 2.0

def random_dir():
            theta = random.uniform(0.0, 2*math.pi)
            return (math.cos(theta), math.sin(theta), 0)


def add_ring(sim, cellType, center, R_ring, ring_width, spacing,
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

            sim.addCell(
                cellType=cellType,
                pos=(x, y, 0),
                dir=random_dir()
            )

def setup(sim):
    # Set biophysics, signalling, and regulation models
    biophys = CLBacterium(sim, jitter_z=False, gamma = 50, max_cells=max_cells)
    sig = GridDiffusion(sim, n_signals, grid_dim, grid_size, grid_orig, D=[Diff_A, Diff_N], initLevels=[ANTIBIOTIC, NUTRIENTS])
    integ = CLCrankNicIntegrator(sim, n_signals, n_species, max_cells, sig, boundcond='nearest')

    # use this file for reg too
    regul = ModuleRegulator(sim, sim.moduleName)
    # Only biophys and regulation
    sim.init(biophys, regul, sig, integ)

    
    
    R_ring = 40.0
    ring_width = 30.0
    spacing = 7.0

    center_S = (-60.0, 0.0)
    center_R = (60.0, 0.0)

    # Only build the half of each ring facing the other colony (theta=0
    # points along +x, theta=pi along -x), which roughly halves the cell
    # count and speeds up the simulation.
    add_ring(sim, 0, center_S, R_ring, ring_width, spacing,
             theta_center=0.0, theta_span=math.pi)
    add_ring(sim, 1, center_R, R_ring, ring_width, spacing,
             theta_center=math.pi, theta_span=math.pi)

    # Specify the initial cell and its location in the simulation
    #sim.addCell(cellType=0, pos=(-3.0,0,0))  # inhibited by the signal
    #sim.addCell(cellType=1, pos=(3.0,0,0))   # sink that removes the signal

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
        const float D_N = 2.0f;
        const float D_A = 10.0f;
        const float k_A = 5.0f;
        const float k_N = 1.0f;

        float A_in = species[0];
        float A = signals[0];

        float N_in = species[1];
        float N = signals[1];
        
        '''



def specRateCL():
    global cl_prefix
    return cl_prefix + '''

        if (cellType==0){
        // Antibiotic
        rates[0] = D_A*(A-A_in)*area/gridVolume;

        // Nutrient: uptake scales with the cell's current growth rate
        rates[1] = D_N*(N-N_in)*area/gridVolume - k_N*growthRate*N_in;
        }

        else {
        // Antibiotic
        rates[0] = D_A*(A-A_in)*area/gridVolume - k_A*A_in;

        // Nutrient: uptake scales with the cell's current growth rate
        rates[1] = D_N*(N-N_in)*area/gridVolume - k_N*growthRate*N_in;
        }
    '''

def sigRateCL():
    global cl_prefix
    return cl_prefix + '''

        if (cellType==0){
        // Antibiotic
        rates[0] = 0;
        // Nutrient
        rates[1] = -D_N*(N-N_in)*area/gridVolume;
        }
    
        else {
        // Antibiotic
        rates[0] = -D_A*(A-A_in)*area/gridVolume;
        // Nutrient
        rates[1] = -D_N*(N-N_in)*area/gridVolume;
        }
    '''


MIN_BRIGHTNESS = 0.4



def update(cells):
    #Iterate through each cell and flag cells that reach target size for division
    v_max = 1.0    #max if no A 

    # Nutrient half-saturation constant, so if N=K_N, growth rate = 0.5*v_max
    K_N = 1

    # Antibiotic half-inhibition constant, so if A=K_A, growth rate = 0.5*v_max
    K_A = 1

    for (id, cell) in cells.items():

        #Intracellular concentrations
        A_in = cell.species[0]
        N_in = cell.species[1]

        # Nutrient-dependent growth rate
        nutrient_factor = N_in / (K_N + N_in)

        if cell.cellType==0:
            # Sensitive strain

            antibiotic_factor = 1 / (1 + A_in/K_A)

            cell.growthRate = (
                 v_max 
                 * nutrient_factor 
                 * antibiotic_factor
            )

            cell.color = [0.6, 0.1, 0.1] 

        else:
            cell.growthRate = v_max * nutrient_factor
            cell.color = [0.1, 0.1, 0.6] 

        if cell.volume > cell.targetVol:
            cell.divideFlag = True

def divide(parent, d1, d2):
    # Specify target cell size that triggers cell division
    d1.targetVol = 2.5 + random.uniform(0.0,0.5)
    d2.targetVol = 2.5 + random.uniform(0.0,0.5)