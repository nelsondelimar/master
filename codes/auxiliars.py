# Title: Auxiliars
# Author: Nelson Ribeiro Filho
# Description: Auxiliars codes to help on some calculations
# Collaborator: Rodrigo Bijani

import numpy
import math
import scipy
import warnings

def my_atan(x, y):
    
    '''
    Return the more stable output for arctan calculation by correcting the 
    value of the angle in arctan2, in order to fix the sign of the tangent.
    '''
    arctan = numpy.arctan2(x, y)
    arctan[x == 0] = 0
    arctan[(x > 0) & (y < 0)] -= numpy.pi
    arctan[(x < 0) & (y < 0)] += numpy.pi
    return arctan

def my_log(x):
    
    ''' 
    Return the value 0 for log(0), once the limits applying in the formula
    tend to 0.
    '''
    
    log = numpy.log(x)
    log[x == 0] = 0
    return log

def my_dot(x,y):
    
    '''
    Return the safe value for the dot product between two vectors.
    '''
    
    return numpy.dot(x,y)

def my_hadamard(x,y):
    
    '''
    Return the safe value for the hadamard product between two vectors.
    '''
    
    return numpy.multiply(x,y)

def my_outer(x,y):
    
    '''
    Return the safe value for the outer product between two vectors.
    '''
    
    return numpy.outer(x,y)

def deg2rad(angle):
    '''
    It converts an angle value in degrees to radian.     
    
    Input:
    angle - float - number or list of angle in degrees    
    Output:
    argument - float - angle in radian    
    '''
    
    # Angle conversion
    argument = (angle/180.)*numpy.pi
    
    # Return the final output
    return argument

def rad2deg(argument):
    '''
    This function converts an angle value in radian to an another value in degrees.
    
    Input:
    argument - float - number or list of angle in radian
    Output:
    angle - float - angle in degrees    
    '''
    
    # Angle conversion
    angle = (argument/numpy.pi)*180.
    # Return the final output
    return angle

def dircos(inc, dec, azm = 0.):
    '''
    This function calculates the cossines projected values on directions using inclination 
    and declination values. Here, we do not considerate an azimuth as a zero value, but as 
    an input value.    
    
    Inputs:
    theta_inc - inclination angle
    theta_dec - declination angle 
    Outputs:
    dirA - projected cossine A
    dirB - projected cossine B
    dirC - projected cossine C    
    '''
    
    # Use the function to convert some values
    incl = deg2rad(inc)
    decl = deg2rad(dec)
    azim = deg2rad(azm)
    # Calculates the projected cossine values
    A = numpy.cos(incl)*numpy.cos(decl - azim)
    B = numpy.cos(incl)*numpy.sin(decl - azim)
    C = numpy.sin(incl)
    
    # Return the final output
    return A, B, C

def regional(intensity, incf, decf):
    '''
    This fucntion computes the projected components of the regional magnetic field in all 
    directions X, Y and Z. This calculation is done by using a cossine projected function, 
    which recieves the values for an inclination, declination and also and azimuth value. 
    It returns all three components for a magnetic field (Fx, Fy e Fz), using a value for 
    the regional field (F) as a reference for the calculation.
    
    Inputs: 
    field - numpy array
        intensity - float - regional magnetic intensity
        incF - float - magnetic field inclination value
        decF - float - magnetic field declination value
    Outputs:
    vecF - numpy array - F componentes along X, Y e Z axis
        
    Ps. All inputs can be arrays when they are used for a set of values.    
    '''
    
    # Computes the projected cossine
    X, Y, Z = dircos(incf, decf,)
    
    # Compute all components
    Fx, Fy, Fz = intensity*X, intensity*Y, intensity*Z
    
    # Set F as array and return the output
    return Fx, Fy, Fz

def noise_normal_dist(data, v0, std):
    '''
    This function adds noise along the input data using a normal Gaussian distribution for each 
    point along the data set.
    If data is a numpy 1D array whit N elements, this function returns a simple length N vector, 
    else it returns a 2D array with NM elements.    
    '''
    
    assert numpy.min(data) <= numpy.mean(data), 'Mean must be greater than minimum'
    assert numpy.max(data) >= numpy.mean(data), 'Maximum must be greater than mean'
    assert std <= 10., 'Noise must not be greater than 1'
    assert std >= 1e-12, 'Noise should not be smaller than 1 micro unit'
    
    # Define the values for size and shape of the data
    size = data.size
    shape = data.shape
    
    # Creat the zero vector such as data
    noise = numpy.zeros_like(data)
    
    # Calculate the local number
    #local = np.abs((data.max() - data.min())*(1e-2))
    
    # Verify if data is a 1D or 2D array
    if data.shape[0] == size or data.shape[1] == size:
        noise = numpy.random.normal(v0, std, size)
    else:
        noise = numpy.random.normal(v0, std, shape)
        
    # Return the final output
    return data + noise

def noise_uniform_dist(data, vmin, vmax):
    '''
    This function adds noise along the input data using a uniform distribution for each point 
    along the data set. 
    '''
    
    assert numpy.min(data) <= numpy.mean(data), 'Mean must be greater than minimum'
    assert numpy.max(data) >= numpy.mean(data), 'Maximum must be greater than mean'
    
    # Define the values for size and shape of the data
    size = data.size
    shape = data.shape
    
    # Creat the zero vector such as data
    noise = numpy.zeros_like(data)
    
    # Calculate the local number
    #local = np.abs((data.max() - data.min())*(1e-2))
    
    # Verify if data is a 1D or 2D array
    if data.shape[0] == size or data.shape[1] == size:
        noise = numpy.random.uniform(vmin, vmax, size = size)
    else:
        noise = numpy.random.uniform(vmin, vmax, size = shape)
        
    # Return the final output
    return data + noise

def residual(observed, predicted):
    '''
    It calculates the residual between the observed data and the calculated predicted data.
    
    Inputs:
    observed - numpy array or list - observed data
    predicted - numpy array or list - predicted data
    
    Outputs:
    norm - numpy array or list - norm data values
    mean - float - mean of all values
    std - float - calculated tandard deviation
    '''
    
    if observed.shape != predicted.shape:
        raise ValueError("All inputs must have same shape!") 
    
    # Calculates the residual
    res = observed - predicted
    # Calculates the mean value of the residual
    mean = numpy.mean(res)
    # Calculates the standard deviation of the residual
    std = numpy.std(res)
    # Calculates the array of norms
    norm = (res - mean)/(std)
    
    # Returns the output final
    return res, norm, mean, std

def theta(inc, dec, u, v):    
    '''
    Return the operators for magnetization and field directions.
    
    Inputs:
    angle - numpy 1D array - inclination and declination
    u - float - number of points in x direction
    v - float - number of points in y direction
    
    Output:
    theta - complex - magnetization projection as a complex number
    '''
    
    # Calculate the modulus for k value. In this case: k = kz
    k = (u**2 + v**2)**(0.5)
    
    # Calcutaing the projections
    x, y, z = dircos(inc, dec) 
    theta = z + ((x*u + y*v)/k)*1j
    
    # Return the final output:
    return theta

def wavenumber(x, y):
    '''
    Return the wavenumbers in X and Y directions
    
    Inputs:
    x - numpy array - coordinates in x directions
    y - numpy array - coordinates in y directions
    
    Output:
    kx - numpy 2D array - calculated wavenumber in x direction
    ky - numpy 2D array - calculated wavenumber in y direction
    '''
    
    # Verify if x and y are 1D or 2D numpy arrays
    if x.shape[0] == x.size or x.shape[1] == x.size:
        dx = x[1] - x[0]
        nx = x.size
    else:
        dx = (x.max() - x.min())/(x.shape[1] - 1)
        nx = x.shape[1]
    
    if y.shape[0] == y.size or y.shape[1] == y.size:
        dy = y[1] - y[0]
        ny = y.size
    else:
        dy = (y.max() - y.min())/(y.shape[0] - 1)
        ny = y.shape[0]
      
    # Compute the values for wavenumber in x and y directions
    c = 2.*numpy.pi
    kx = c*numpy.fft.fftfreq(nx, dx)
    ky = c*numpy.fft.fftfreq(ny, dy)    
    
    # Return the final output
    return numpy.meshgrid(kx, ky)

def rotation_x(angle):
    '''    
    It returns the rotation matrix given a (x, y, z) point at x direction,
    
    Inputs: 
    angle - numpy float - angle of rotation     
    '''
     
    #assert angle <= 360, 'Angulo em graus deve ser menor ou igual a 360'
    #assert angle >= 0, 'Angulo em graus deve ser maior ou igual a 0'
     
    argument = (angle/180.)*numpy.pi
    c = numpy.cos(argument)
    s = numpy.sin(argument)
  
    return numpy.array([[1., 0., 0.,],[0., c, s],[0., -s, c]])

def rotation_y(angle):
    '''     
    It returns the rotation matrix given a (x, y, z) point at y direction,
    
    Inputs: 
    angle - numpy float - angle of rotation

    Output:
    ry - numpy array 2D - matrix of rotation at y direction    
    '''
     
    #assert angle <= 360, 'Angulo em graus deve ser menor ou igual a 360'
    #assert angle >= 0, 'Angulo em graus deve ser maior ou igual a 0'
     
    argument = (angle/180.)*numpy.pi
    c = numpy.cos(argument)
    s = numpy.sin(argument)
     
    return numpy.array([[c, 0., s,],[0., 1., 0.],[-s, 0., c]])

def rotation_z(angle):
    '''    
    It returns the rotation matrix given a (x, y, z) point at z direction,
    
    Inputs: 
    angle - numpy float - angle of rotation
     
    Output:
    rz - numpy array 2D - matrix of rotation at z direction
    '''
     
    #assert angle <= 360, 'Angulo em graus deve ser menor ou igual a 360'
    #assert angle >= 0, 'Angulo em graus deve ser maior ou igual a 0'
     
    argument = (angle/180.)*numpy.pi
    c = numpy.cos(argument)
    s = numpy.sin(argument)

    return numpy.array([[c, s, 0.,],[-s, c, 0.],[0., 0., 1.]])

def rotate3D_xyz(x, y, z, angle, direction = 'z'):
    '''   
    It returns the rotated plane x-y along z-axis by default.
    If angle is positive, the rotation in counterclockwise direction; 
    otherwise is clockwise direction.
    
    Inputs:
    x, y, z - numpy arrays - coordinate points
    angle - float - angle of rotation
    direction - string - direction
    
    Outputs:
    xr, yr, zr - numpy arrays - new rotated coordinate points
    '''
    
    # Size condition
    #if x.shape != y.shape:
    #    raise ValueError("All inputs must have same shape!")
    # Matrix rotation in x-y-or-z direction
    if direction == 'x':
        rot = rotation_x(angle)
    if direction == 'y':
        rot = rotation_y(angle)
    if direction == 'z':
        rot = rotation_z(angle)
        
    # Create the matrix
    mat = numpy.vstack([x, y, z]).T
    # Create the zero matrix
    res = numpy.zeros_like(mat)
    
    # Calculate the rotated coordinates
    for k, i in enumerate(mat):
        res[k,:] = numpy.dot(rot, i)
    
    # New coordinates
    xr = res[:,0]
    yr = res[:,1]
    zr = res[:,2]
    
    # Return the final output
    return xr, yr, zr
