# Title: Auxiliars
# Author: Nelson Ribeiro Filho
# Description: Auxiliars codes to help on some calculations
# Collaborator: Rodrigo Bijani

import numpy as np
import scipy as sp
import warnings

def deg2rad(angle):
    '''
    This function converts an angle value in degrees to an another value in radian.     
    
    Input:
    angle - float - angle in degrees    
    Output:
    argument - float - angle in radian    
    '''
    
    # Condition for the calculation
    if angle > 360.:
        r = angle//360
        angle = angle - r*360
    
    # Angle conversion
    argument = (angle/180.)*np.pi
    
    # Return the final output
    return argument

def rad2deg(argument):
    '''
    This function converts an angle value in radian to an another value in degrees.
    
    Input:
    argument - float - angle in radian
    Output:
    angle - float - angle in degrees    
    '''
    
    # Check the input value for an angle
    assert type(argument) is float, 'Angle value must be decimal!'
    # Angle conversion
    angle = (argument/np.pi)*180.
    # Return the final output
    return angle

def dircos(inc, dec, azm = None):
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
        
    # Stablishing the input conditions
    if azm is None:        
        # Calculates the projected cossine values
        A = np.cos(incl)*np.cos(decl)
        B = np.cos(incl)*np.sin(decl)
        C = np.sin(incl)
    else:
        azim = deg2rad(azm)
        # Calculates the projected cossine values
        A = np.cos(incl)*np.cos(decl - azim)
        B = np.cos(incl)*np.sin(decl - azim)
        C = np.sin(incl)
    
    # Return the final output
    return A, B, C

def regional(F, field):
    '''
    This fucntion computes the projected components of the regional magnetic field in all 
    directions X, Y and Z. This calculation is done by using a cossine projected function, 
    which recieves the values for an inclination, declination and also and azimuth value. 
    It returns all three components for a magnetic field (Fx, Fy e Fz), using a value for 
    the regional field (F) as a reference for the calculation.
    
    Inputs: 
    field - numpy array
        valF - float - regional magnetic field value
        incF - float - magnetic field inclination value
        decF - float - magnetic field declination value
    Outputs:
    vecF - numpy array - F componentes along X, Y e Z axis
        
    Ps. All inputs can be arrays when they are used for a set of values.    
    '''
    
    assert field[0] != 0., 'Value of the regional magnetic field must be nonzero!'
        
    # Computes the projected cossine
    X, Y, Z = dircos(field[0], field[1])
    
    # Compute all components
    Fx, Fy, Fz = F*X, F*Y, F*Z
    
    # Set F as array and return the output
    return [Fx, Fy, Fz]

def addnoise(data, v0, std):
    '''
    This function adds noise along the input data using a normal Gaussian distribution for each 
    point along the data set.
    If data is a numpy 1D array whit N elements, this function returns a simple length N vector, 
    else it returns a 2D array with NM elements.    
    '''
    
    assert np.min(data) <= np.mean(data), 'Mean must be greater than minimum'
    assert np.max(data) >= np.mean(data), 'Maximum must be greater than mean'
    assert std <= 10., 'Noise must not be greater than 1'
    assert std >= 1e-12, 'Noise should not be smaller than 1 micro unit'
    
    # Define the values for size and shape of the data
    size = data.size
    shape = data.shape
    
    # Creat the zero vector such as data
    noise = np.zeros_like(data)
    
    # Calculate the local number
    #local = np.abs((data.max() - data.min())*(1e-2))
    
    # Verify if data is a 1D or 2D array
    if data.shape[0] == size or data.shape[1] == size:
        noise = np.random.normal(loc = v0, scale = std, size = size)
    else:
        noise = np.random.normal(loc = v0, scale = std, size = shape)
        
    # Return the final output
    return data + noise

def my_atan(x, y):
    
    '''
    Return the more stable output for arctan calculation by correcting the 
    value of the angle in arctan2, in order to fix the sign of the tangent.
    '''
    arctan = np.arctan2(x, y)
    arctan[x == 0] = 0
    arctan[(x > 0) & (y < 0)] -= np.pi
    arctan[(x < 0) & (y < 0)] += np.pi
    return arctan

def my_log(x):
    
    ''' 
    Return the value 0 for log(0), once the limits applying in the formula
    tend to 0.
    '''
    
    log = np.log(x)
    log[x == 0] = 0
    return log

def theta(angle, u, v):
    
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
    
    # Defines inclination and declination:
    inc, dec = angle[0], angle[1]
    
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
    c = 2.*np.pi
    kx = c*np.fft.fftfreq(nx, dx)
    ky = c*np.fft.fftfreq(ny, dy)    
    
    # Return the final output
    return np.meshgrid(kx, ky)