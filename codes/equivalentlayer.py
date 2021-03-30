from __future__ import division
import numpy
from codes import auxiliars, grids, kernel

# Building the Classical Equivalent Layer Technique
def my_layer(area, shape, level):
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
    xo, yo, zo = grids.my_regular(area, shape, level)
    
    # Create a empty list
    layer = []
        
    # Includes all source points
    for i in range(len(xo)):
        layer.append([xo[i],yo[i],zo[i],R])
    
    # Return the final output
    return layer

# Applying for gravity data
def my_gz_layer(xo, yo, zo, layer):
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
        mat[:,i] = kernel.my_kernelz(xo, yo, zo, j)

    mat *= g * si2mGal
    
    # Return the final output
    return mat

def my_gxyz_layer(xo, yo, zo, layer):
    '''
    It calculates the sensitivity matrix for a gravity case as horizontal and vertical derivatives 
    of the vertical gravitational component. It must receives all observation points and the layer 
    model as a list.
    
    Inputs:
    xo, yo, zo - numpy array - observation points
    layer - numpy list - created layer as a list
    
    Output:
    gzx - numpy matrix - first order x-derivative of gz data
    gzy - numpy matrix - first order y-derivative of gz data
    gzz - numpy matrix - first order z-derivative of gz data
    '''
    # Constants
    g = 0.00000006673
    si2mGal = 100000.
    
    # Number of points:
    n = len(xo)
    m = len(layer)
    
    # Create all zero matrix
    gzx = numpy.zeros((n, m))
    gzy = numpy.zeros((n, m))
    gzz = numpy.zeros((n, m))
    
    # Calculates all kernels
    for i,j in enumerate(layer):
        gzx[:,i] = kernel.my_kernelxz(xo, yo, zo, j)
        gzy[:,i] = kernel.my_kernelyz(xo, yo, zo, j)
        gzz[:,i] = kernel.my_kernelzz(xo, yo, zo, j)

    gzx *= g * si2mGal
    gzy *= g * si2mGal
    gzz *= g * si2mGal
    
    # Return the final output
    return gzx, gzy, gzz

def my_fitdata_grav(dataset, datashape, layermodel, layershape, regulator):
    '''
    It returns the predicted data by using classical equivalent layer technique. This function must receive 
    all data as a list with all positions for x, y and z, and also the potential data. It receives the 
    shape of the data, the model as an equivalent layer and the value for the regularization pvec.    
    
    Inputs:
    datasets - numpy list - x, y and z positions and gravity data
    datashape - tuple - shape of the input data
    layermodel - list - values for created equivalent layer
    layershape - tuple - shape of the equivalent layer
    regulator - float - zero order Tikhonov regulator
    
    Output:
    pvec - numpy array - pvecs vector
    predicted - numpy array - predicted total field data
    
    '''
    
    # Datasets = [xobs, yobs, zobs, totalfield]
    xp = dataset[0]
    yp = dataset[1]
    zp = dataset[2]
    gz = dataset[3]
    
    # Define the number of observations and depth sources
    N = datashape[0]*datashape[1]
    M = layershape[0]*layershape[1]
    
    # Computes the sensitivity matrix
    mat = my_gz_layer(xp, yp, zp, layermodel)

    # Case1: Overdetermined - Number of observations are greater or equal than the number of depth sources
    if N >= M: 
        I = numpy.identity(M)
        trace = numpy.trace(numpy.dot(mat.T, mat))/M
        pvec = numpy.linalg.solve(numpy.dot(mat.T, mat) + regulator*trace*I, numpy.dot(mat.T, gz))
    # Case2: Underterminated - Number of observations are less than the number of depth sources
    else:
        I = numpy.identity(N)
        trace = numpy.trace(numpy.dot(mat.T, mat))/N
        aux = numpy.linalg.solve(numpy.dot(mat, mat.T) + regulator*trace*I, gz)
        pvec = numpy.dot(mat.T, aux)
        
    # Calculates the predicted total field anomaly data
    predicted = numpy.dot(mat, pvec)

    # Return the final output
    return pvec, predicted

# Applying for magnetic data - Total field anomaly and reduction to Pole
def my_totalfield_layer(xo, yo, zo, layer, inc, dec, incs, decs):
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
    fx, fy, fz = auxiliars.my_regional(1., inc, dec)
    mx, my, mz = auxiliars.my_regional(1., incs, decs)
    
    # Dealing with all kernels at all directions
    for i,j in enumerate(layer):
        phi_xx = kernel.my_kernelxx(xo, yo, zo, j)
        phi_yy = kernel.my_kernelyy(xo, yo, zo, j)
        phi_zz = kernel.my_kernelzz(xo, yo, zo, j)
        phi_xy = kernel.my_kernelxy(xo, yo, zo, j)
        phi_xz = kernel.my_kernelxz(xo, yo, zo, j)
        phi_yz = kernel.my_kernelyz(xo, yo, zo, j)
        mat[:,i] = fx*phi_xx*mx + fx*phi_xy*my + fx*phi_xz*mz + \
                   fy*phi_xy*mx + fy*phi_yy*my + fy*phi_yz*mz + \
                   fz*phi_xz*mx + fz*phi_yz*my + fz*phi_zz*mz

    mat *= cm * t2nT
    
    # Return the final output
    return mat

def my_fitdata_mag(dataset, datashape, layermodel, layershape, regulator, inc, dec, incl = None, decl = None):
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
    pvec - numpy array - pvecs vector
    predicted - numpy array - predicted total field data
    
    '''
    
    # Define the type of magnetization
    if incl == None or decl == None:
        incl = inc
        decl = dec
        
    # Datasets = [xobs, yobs, zobs, totalfield]
    xp = dataset[0]
    yp = dataset[1]
    zp = dataset[2]
    tf = dataset[3]
    
    # Define the number of observations and depth sources
    N = datashape[0]*datashape[1]
    M = layershape[0]*layershape[1]
    
    # Computes the sensitivity matrix
    mat = my_totalfield_layer(xp, yp, zp, layermodel, inc, dec, incl, decl)

    # Case1: Overdetermined - Number of observations are greater or equal than the number of depth sources
    if N >= M: 
        I = numpy.identity(M)
        trace = numpy.trace(numpy.dot(mat.T, mat))/M
        pvec = numpy.linalg.solve(numpy.dot(mat.T, mat) + regulator*trace*I, numpy.dot(mat.T, tf))
    # Case2: Underterminated - Number of observations are less than the number of depth sources
    else:
        I = numpy.identity(N)
        trace = numpy.trace(numpy.dot(mat.T, mat))/N
        aux = numpy.linalg.solve(numpy.dot(mat, mat.T) + regulator*trace*I, tf)
        pvec = numpy.dot(mat.T, aux)
        
    # Calculates the predicted total field anomaly data
    predicted = numpy.dot(mat, pvec)

    # Return the final output
    return pvec, predicted

def my_rtp_layer(datasets, datashape, layermodel, layershape, regulator, inc, dec, incl = None, decl = None):
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
    if incl == None or decl == None:
        incl = inc
        decl = dec
        
    # Datasets = [xobs, yobs, zobs, totalfield]
    xp = datasets[0]
    yp = datasets[1]
    zp = datasets[2]
    tf = datasets[3]
    
    # Define the number of observations and depth sources
    N = datashape[0]*datashape[1]
    M = layershape[0]*layershape[1]
    
    # Computes the sensitivity matrix
    mat = my_totalfield_layer(xp, yp, zp, layermodel, inc, dec, incl, decl)
    
    # Case1: Overdetermined - Number of observations are greater or equal than the number of depth sources
    if N >= M: 
        I = numpy.identity(M)
        trace = numpy.trace(numpy.dot(mat.T, mat))/M
        vec = numpy.linalg.solve(numpy.dot(mat.T, mat) + regulator*trace*I, numpy.dot(mat.T, tf))
    # Case2: Underterminated - Number of observations are less than the number of depth sources
    else:
        I = numpy.identity(N)
        trace = numpy.trace(numpy.dot(mat.T, mat))/N
        aux = numpy.linalg.solve(numpy.dot(mat, mat.T) + regulator*trace*I, tf)
        vec = numpy.dot(mat.T, aux)
        
    # Calculates the predicted total field anomaly data
    pred = numpy.dot(mat, vec)
    # Create the new matrix for reduction to Pole
    rtp_mat = my_totalfield_layer(xp, yp, zp, layermodel, 90., 0., 90., 0.)
    # Calculates the reduction to Pole
    rtp = numpy.dot(rtp_mat, vec)    
    
    # Return the final output
    return rtp

def my_rte_layer(datasets, datashape, layermodel, layershape, regulator, incf, decf, inceql = None, deceql = None):
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
    if incl == None or decl == None:
        incl = inc
        decl = dec
        
    # Datasets = [xobs, yobs, zobs, totalfield]
    xp = datasets[0]
    yp = datasets[1]
    zp = datasets[2]
    tf = datasets[3]
    
    # Define the number of observations and depth sources
    N = datashape[0]*datashape[1]
    M = layershape[0]*layershape[1]
    
    # Computes the sensitivity matrix
    mat = my_totalfield_layer(xp, yp, zp, layermodel, inc, dec, incl, decl)
    
    # Case1: Overdetermined - Number of observations are greater or equal than the number of depth sources
    if N >= M: 
        I = numpy.identity(M)
        trace = numpy.trace(numpy.dot(mat.T, mat))/M
        vec = numpy.linalg.solve(numpy.dot(mat.T, mat) + regulator*trace*I, numpy.dot(mat.T, tf))
    # Case2: Underterminated - Number of observations are less than the number of depth sources
    else:
        I = numpy.identity(N)
        trace = numpy.trace(numpy.dot(mat.T, mat))/N
        aux = numpy.linalg.solve(numpy.dot(mat, mat.T) + regulator*trace*I, tf)
        vec = numpy.dot(mat.T, aux)
        
    # Calculates the predicted total field anomaly data
    pred = numpy.dot(mat, vec)
    # Create the new matrix for reduction to Pole
    rteq_mat = my_totalfield_layer(xp, yp, zp, layermodel, 45., 0., 45., 0.)
    # Calculates the reduction to Pole
    rteq = numpy.dot(rteq_mat, vec)    
    # Return the final output
    return rteq