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
    if level < 0. or level == 0.:
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


def mat_mag_tfa(xo, yo, zo, layer, inc, dec, incs, decs):
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
    
    # Number of points:
    n = len(xo)
    m = len(layer)

    # Create the zero-matrix
    mat = numpy.zeros((n,m))
    
    # Calculate the projections in x, y and z directions
    fx, fy, fz = regional(1., inc, dec)
    mx, my, mz = regional(1., incs, decs)
    
    # Dealing with all kernels at all directions
    for i,j in enumerate(layer):
        phi_xx = kernel.kernelxx(xo, yo, zo, j)
        phi_yy = kernel.kernelyy(xo, yo, zo, j)
        phi_zz = kernel.kernelzz(xo, yo, zo, j)
        phi_xy = kernel.kernelxy(xo, yo, zo, j)
        phi_xz = kernel.kernelxz(xo, yo, zo, j)
        phi_yz = kernel.kernelyz(xo, yo, zo, j)
        mat[:,i] = fx*phi_xx*mx + fx*phi_xy*my + fx*phi_xz*mz + \
                   fy*phi_xy*mx + fy*phi_yy*my + fy*phi_yz*mz + \
                   fz*phi_xz*mx + fz*phi_yz*my + fz*phi_zz*mz

    mat *= cm * t2nT
    
    # Return the final output
    return mat

def mat_grav_gz(xo, yo, zo, layer):
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
    
    # Number of points:
    n = len(xo)
    m = len(layer)
    
    # Create the zero matrix
    mat = numpy.zeros((n, m))
    
    # Calculates the kernel
    for i,j in enumerate(layer):
        mat[:,i] = kernel.kernelz(xo, yo, zo, j)

    mat *= g * si2mGal
    
    # Return the final output
    return mat

def mat_grav_gz_xyz(xo, yo, zo, layer):
    '''
    It calculates the sensitivity matrix for a gravity case as horizontal and vertical derivatives 
    of the vertical gravitational component. It must receives all observation points and the layer 
    model as a list.
    
    Inputs:
    xo, yo, zo - numpy array - observation points
    layer - numpy list - created layer as a list
    
    Output:
    mat_gzx - numpy matrix - first order x-derivative of gz data
    mat_gzy - numpy matrix - first order y-derivative of gz data
    mat_gzz - numpy matrix - first order z-derivative of gz data
    '''
    # Constants
    g = 0.00000006673
    si2mGal = 100000.
    
    # Number of points:
    n = len(xo)
    m = len(layer)
    
    # Create all zero matrix
    mat_gzx = numpy.zeros((n, m))
    mat_gzy = numpy.zeros((n, m))
    mat_gzz = numpy.zeros((n, m))
    
    # Calculates all kernels
    for i,j in enumerate(layer):
        mat_gzx[:,i] = kernel.kernelxz(xo, yo, zo, j)
        mat_gzy[:,i] = kernel.kernelyz(xo, yo, zo, j)
        mat_gzz[:,i] = kernel.kernelzz(xo, yo, zo, j)

    mat_gzx *= g * si2mGal
    mat_gzy *= g * si2mGal
    mat_gzz *= g * si2mGal
    
    # Return the final output
    return mat_gzx, mat_gzy, mat_gzz

def fit_layer(datasets, datashape, layermodel, layershape, regulator, inc, dec, inclayer = None, declayer = None):
    '''
    It returns the predicted data by using classical equivalent layer technique. This function must receive 
    all data as a list with all positions for x, y and z, and also the potential data. It receives the 
    shape of the data, the model as an equivalent layer and the values of inclination and declination as 
    well, for both field and depth sources.    
    
    Inputs:
    datasets - numpy list - x, y and z positions and total field data
    datashape - tuple - shape of the input data
    layermodel - list - values for created equivalent layer
    layershape - tuple - shape of the equivalent layer
    regulator - float - zero order Tikhonov regulator
    inc - float - inclination of the geomagnetic field
    dec - float - declination of the geomagnetic field
    inclayer - float - inclination of all depth sources
    declayer - float - declination of all depth sources
    
    Output:
    parameter - numpy array - parameters vector
    predicted - numpy array - predicted total field data
    
    '''
    
    # Define the type of magnetization
    if inclayer == None or declayer == None:
        inclayer = inc
        declayer = dec
        
    # Datasets = [xobs, yobs, zobs, totalfield]
    xp = datasets[0]
    yp = datasets[1]
    zp = datasets[2]
    tf = datasets[3]
    
    # Define the number of observations and depth sources
    N = datashape[0]*datashape[1]
    M = layershape[0]*layershape[1]
    
    # Computes the sensitivity matrix
    matA = mat_mag_tfa(xp, yp, zp, layermodel, inc, dec, inclayer, declayer)

    # Case1: Overdetermined - Number of observations are greater or equal than the number of depth sources
    if N >= M: 
        I = numpy.identity(M)
        trace = numpy.trace(numpy.dot(matA.T, matA))/M
        vec = numpy.linalg.solve(numpy.dot(matA.T, matA) + regulator*trace*I, numpy.dot(matA.T, tf))
    # Case2: Underterminated - Number of observations are less than the number of depth sources
    else:
        I = numpy.identity(N)
        trace = numpy.trace(numpy.dot(matA.T, matA))/N
        aux = numpy.linalg.solve(numpy.dot(matA, matA.T) + regulator*trace*I, tf)
        vec = numpy.dot(matA.T, aux)
        
    # Calculates the predicted total field anomaly data
    predicted = numpy.dot(matA, vec)

    # Return the final output
    return vec, predicted

def rtp_layer(datasets, datashape, layermodel, layershape, regulator, incf, decf, inceql = None, deceql = None):
    '''
    It returns the reduce to Pole data by using the equivalent layer technique. This functions 
    must receives all data as a list with all positions for x, y and z, and also the potential 
    data. It receives the shape of the data, the model as an equivalent layer and the values of
    inclination and declination as well, for both field and depth sources.    
    
    Inputs:
    datasets - numpy list - x, y and z positions and total field data
    datashape - tuple - shape of the input data
    layermodel - list - values for created equivalent layer
    layershape - tuple - shape of the equivalent layer
    incf - float - inclination of the geomagnetic field
    decf - float - declination of the geomagnetic field
    inceql - float - inclination of all depth sources
    deceql - float - declination of all depth sources
    
    Output:
    rtp - numpy array - reduce to Pole data
    
    '''
    
    # Define the type of magnetization
    if inceql == None or deceql == None:
        inceql = incf
        deceql = decf
        
    # Datasets = [xobs, yobs, zobs, totalfield]
    xp = datasets[0]
    yp = datasets[1]
    zp = datasets[2]
    tf = datasets[3]
    
    # Define the number of observations and depth sources
    N = datashape[0]*datashape[1]
    M = layershape[0]*layershape[1]
    
    # Computes the sensitivity matrix
    matA = mat_mag_tfa(xp, yp, zp, layermodel, incf, decf, inceql, deceql)
    
    # Case1: Overdetermined - Number of observations are greater or equal than the number of depth sources
    if N >= M: 
        I = numpy.identity(M)
        trace = numpy.trace(numpy.dot(matA.T, matA))/M
        vec = numpy.linalg.solve(numpy.dot(matA.T, matA) + regulator*trace*I, numpy.dot(matA.T, tf))
    # Case2: Underterminated - Number of observations are less than the number of depth sources
    else:
        I = numpy.identity(N)
        trace = numpy.trace(numpy.dot(matA.T, matA))/N
        aux = numpy.linalg.solve(numpy.dot(matA, matA.T) + regulator*trace*I, tf)
        vec = numpy.dot(matA.T, aux)
        
    # Calculates the predicted total field anomaly data
    tf_pred = numpy.dot(matA, vec)
    
    # Create the new matrix for reduction to Pole
    mat_rtp = mat_mag_tfa(xp, yp, zp, layermodel, 90., 0., 90., 0.)
    # Calculates the reduction to Pole
    rtp = numpy.dot(mat_rtp, vec)    
    
    # Return the final output
    return rtp
