# -----------------------------------------------------------------------------------
# Title: Equivalent Layer
# Author: Nelson Ribeiro Filho
# Description: Classical equivalent layer module which includes the sensitivity matrix
# Collaborator: Rodrigo Bijani
# -----------------------------------------------------------------------------------
import numpy
import kernel
from auxiliars import regional
from grids import regular_grid

def layer(area, shape, level):
    '''
    It generates a list with all 3D spheres position.
    
    Inputs:
    area - numpy list - minimum and maximum values in x and y directions
    shape - tuple - shape of the gridded data 
    level - float - layer depth
    
    Output:
    layer - numpy list - all x, y and z positions and the radius value
    '''
    
    # Condition for the level
    if level < 0 or level == 0:
        raise ValueError('Depth of the layer must have be positive or non-null')

    # Calculates the radius of sphere
    R = (3./(4.*numpy.pi))**(1./3.)
    
    # Grid all points
    xo, yo, zo = regular_grid(area, shape, level)
    
    # Create a empty list
    layer = []
    
    # Includes all source points
    for i in range(len(xo)):
        layer.append([xo[i],yo[i],zo[i],R])
    
    # Return the final output
    return layer


def mat_mag(xo, yo, zo, layer, inc, dec, incs, decs):
    '''
    It calculates the sensitivity matrix for a magnetic case as total field anomaly. It must receives 
    all observation points and the layer model as a list. It also recieves all values of inclination
    and declination for the magnetic field and the magnetic source.
    
    Inputs:
    xo, yo, zo - numpy array - observation points
    layer - numpy list - created layer as a list
    inc - float - magnetic field inclination
    dec - float - magnetic field declination
    incs - float - source inclination
    decs - float - source declination
    
    Output:
    mat - numpy matrix - computed sensitivity matrix
    '''
    
    # Constants
    cm = 1.e-7
    t2nT = 1.e9
    
    # Number of points
    n = len(xo)
    # Create the zero-matrix
    mat = numpy.zeros((n,n))
    
    # Calculate the projections in x, y and z directions
    fx, fy, fz = regional(1., incs, decs)
    mx, my, mz = regional(1., inc, dec)
    
    # Dealing with all kernels at all directions
    for i,m in enumerate(layer):
        phi_xx = kernel.kernelxx(xo, yo, zo, m)
        phi_yy = kernel.kernelyy(xo, yo, zo, m)
        phi_zz = kernel.kernelzz(xo, yo, zo, m)
        phi_xy = kernel.kernelxy(xo, yo, zo, m)
        phi_xz = kernel.kernelxz(xo, yo, zo, m)
        phi_yz = kernel.kernelyz(xo, yo, zo, m)
        mat[:,i] = fx*phi_xx*mx + fx*phi_xy*my + fx*phi_xz*mz + \
                   fy*phi_xy*mx + fy*phi_yy*my + fy*phi_yz*mz + \
                   fz*phi_xz*mx + fz*phi_yz*my + fz*phi_zz*mz

    mat *= cm * t2nT
    
    # Return the final output
    return mat

def mat_gravity(xo, yo, zo, layer):
    '''
    It calculates the sensitivity matrix for a gravity case as vertical gravitational component. It must 
    receives all observation points and the layer model as a list.
    
    Inputs:
    xo, yo, zo - numpy array - observation points
    layer - numpy list - created layer as a list
    
    Output:
    mat - numpy matrix - computed sensitivity matrix
    '''
    # Constants
    g = 0.00000006673
    si2mGal = 100000.
    
    # Number of points
    n = len(xo)
    # Create the zero matrix
    mat = numpy.zeros((n, n))
    
    # Calculates the kernel
    for i,m in enumerate(layer):
        mat[:,i] = kernel.kernelz(xo, yo, zo, m)

    mat *= g * si2mGal
    
    # Return the final output
    return mat