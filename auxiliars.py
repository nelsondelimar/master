# -----------------------------------------------------------------------------------
# Title: Auxiliars
# Author: Nelson Ribeiro Filho
# Description: Auxiliars codes for using Gravmag Codes
# Collaborator: Rodrigo Bijani
# -----------------------------------------------------------------------------------

import numpy as np # Numpy library
import scipy as sp
import pandas as pd

# Auxiliar number 01
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

# Auxiliar number 02
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

# Auxiliar number 03
def dircos(inc, dec, azi):
    
    '''
    This function calculates the cossines projected values on directions using inclination 
    and declination values. Here, we do not considerate an azimuth as a zero value, but as 
    an input value.    
    
    Inputs:
    theta_inc - inclination angle
    theta_dec - declination angle
    theta_azi - azimuth angle    
    Outputs:
    dirA - projected cossine A
    dirB - projected cossine B
    dirC - projected cossine C    
    '''
    
    # Use the function to convert some values
    incl = deg2rad(inc)
    decl = deg2rad(dec)
    azim = deg2rad(azi)
    
    # Calculates the projected cossine values
    A = np.cos(incl)*np.cos(decl - azim)
    B = np.cos(incl)*np.sin(decl - azim)
    C = np.sin(incl)
    
    # Return the final output
    return A, B, C

def regional(field):
    
    '''
    This fucntion computes the projected components of the regional magnetic field in all directions X, Y and Z. This calculation is done by using a cossine projected function, which recieves the values for an inclination, declination and also and azimuth value. It returns all three components for a magnetic field (Fx, Fy e Fz), using a value for the regional field (F) as a reference for the calculation.
    
    Inputs: 
    field - numpy array
        valF - float - regional magnetic field value
        incF - float - magnetic field inclination value
        decF - float - magnetic field declination value
        aziF - float - magnetic field azimuth value    
    Outputs:
    vecF - numpy array - F componentes along X, Y e Z axis
        
    Ps. All inputs can be arrays when they are used for a set of values.    
    '''
    
    #assert field[0] != 0., 'Value of the regional magnetic field must be nonzero!'
        
    # Setting the value for the magnetic field
    F = field[0]
    
    # Computes the projected cossine
    X, Y, Z = dircos(field[1], field[2], field[3])
    
    # Compute all components
    Fx = F*X
    Fy = F*Y
    Fz = F*Z
    
    # Set the F values as an array
    vecF =[Fx, Fy, Fz]
    # Return the final output
    return vecF

def addnoise(data, std):
    
    '''
    This function adds noise along the input data using a normal Gaussian distribution for each 
    point along the data set.
    If data is a numpy 1D array whit N elements, this function returns a simple length N vector, 
    else it returns a 2D array with NM elements.    
    '''
    
    assert np.min(data) <= np.mean(data), 'Mean must be greater than minimum'
    assert np.max(data) >= np.mean(data), 'Maximum must be greater than mean'
    assert std <= 1., 'Noise must not be greater than 1'
    assert std >= 1e-3, 'Noise should not be smaller than 0.001'
    
    local = int((data.max() - data.min()))*(1e-2)
    noise = np.random.normal(loc = local, scale = std, size = data.shape)
    return data + noise

#def noise_grav(data, std):