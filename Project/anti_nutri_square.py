import random
from CellModeller.Regulation.ModuleRegulator import ModuleRegulator
from CellModeller.Biophysics.BacterialModels.CLBacterium import CLBacterium
from CellModeller.GUI import Renderers
import numpy
import math

from CellModeller.Signalling.GridDiffusion import GridDiffusion #add
from CellModeller.Integration.CLCrankNicIntegrator import CLCrankNicIntegrator #add

gs = 4
max_cells = 20*int(1e4)

#Specify parameter for solving diffusion dynamics #Add
grid_dim = (80, 80, 8) # dimension of diffusion space, unit = number of grid
grid_size = (gs, gs, gs) # grid size
grid_orig = (-160, -160, -8) # where to place the diffusion space onto simulation space

n_signals = 2
n_species = 2

lambda_s = 1.93  
lambda_r = 1.93

    
ANTIBIOTIC = 30.0
NUTRIENTS = 10.0

Diff_A = 100.0
Diff_N = 2.0

def random_dir():
            theta = random.uniform(0.0, 2*math.pi)
            return (math.cos(theta), math.sin(theta), 0)



def setup(sim):
    # Set biophysics, signalling, and regulation models
    biophys = CLBacterium(sim, jitter_z=False, gamma = 50, max_cells=max_cells)
    sig = GridDiffusion(sim, n_signals, grid_dim, grid_size, grid_orig, D=[Diff_A, Diff_N], initLevels=[ANTIBIOTIC, NUTRIENTS])
    integ = CLCrankNicIntegrator(sim, n_signals, n_species, max_cells, sig, boundcond='nearest')

    # use this file for reg too
    regul = ModuleRegulator(sim, sim.moduleName)
    # Only biophys and regulation
    sim.init(biophys, regul, sig, integ)

    x_outer, x_inner = 35, 3
    spacing = 7

    y_range = numpy.arange(-40, 40, spacing)
    x_range1 = numpy.arange(-x_outer, -x_inner, spacing)
    x_range2 = numpy.arange(x_inner, x_outer, spacing)


    for y in y_range:
        for x in x_range1:
            sim.addCell(
                cellType=0,
                pos=(x, y, 0),
                dir=random_dir()
            )

    for y in y_range:
        for x in x_range2:
            sim.addCell(
                cellType=1,
                pos=(x, y, 0),
                dir=random_dir()
            )


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
        const float k_A = 10.0f;
        const float k_N = 1.0f;

        float A_in = species[0];
        float A = signals[0];

        float N_in = species[1];
        float N = signals[1];
        
        '''


def specRateCL():
    global cl_prefix
    # Remove diffusion in and out of cells

    return cl_prefix + '''

        if (cellType==0){
        // Antibiotic
        rates[0] = (A-A_in);                            // D_A*(A-A_in)*area/gridVolume

        // Nutrient: uptake scales with the cell's current growth rate
        rates[1] = (N-N_in) - k_N*growthRate*N_in;      // D_N*(N-N_in)*area/gridVolume - k_N*growthRate*N_in;
        }

        else {
        // Antibiotic
        rates[0] = (A-A_in) - k_A*A_in;                 // D_A*(A-A_in)*area/gridVolume - k_A*A_in;

        // Nutrient: uptake scales with the cell's current growth rate
        rates[1] = (N-N_in) - k_N*growthRate*N_in;      // D_N*(N-N_in)*area/gridVolume - k_N*growthRate*N_in;
        }
    '''

def sigRateCL():
    global cl_prefix
    return cl_prefix + '''

        if (cellType==0){
        // Antibiotic
        rates[0] = 0;
        // Nutrient
        rates[1] = - (N-N_in);                           // - D_N*(N-N_in)*area/gridVolume;
        }
    
        else {
        // Antibiotic
        rates[0] = - (A-A_in);                          // - D_A*(A-A_in)*area/gridVolume;
        // Nutrient
        rates[1] = - (N-N_in);                          // - D_N*(N-N_in)*area/gridVolume;
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

    # Nutrient half-saturation constant, so if N=K_N, growth rate = 0.5*v_max
    K_N = 1

    # Antibiotic half-inhibition constant, so if A=IC_50, growth rate = 0.5*v_max
    IC_50 = 2.39

    for (id, cell) in cells.items():

        #Intracellular concentrations
        A_in = cell.species[0]
        N_in = cell.species[1]

        nutrient_factor = N_in / (K_N + N_in)

        if cell.cellType==0:
            antibiotic_factor = antibiotic_gr(A_in, IC_50=IC_50)
            cell.growthRate = (
                 lambda_s
                 * nutrient_factor 
                 * antibiotic_factor
            )

            cell.color = [0.6, 0.1, 0.1] 

        else:
            cell.growthRate = lambda_r * nutrient_factor
            cell.color = [0.1, 0.1, 0.6] 

        if cell.volume > cell.targetVol:
            cell.divideFlag = True

def divide(parent, d1, d2):
    # Specify target cell size that triggers cell division
    d1.targetVol = 2.5 + random.uniform(0.0,0.5)
    d2.targetVol = 2.5 + random.uniform(0.0,0.5)