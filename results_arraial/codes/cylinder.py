# --------------------------------------------------------------------------------------------------
# Author: Nelson Ribeiro Filho
# Description: Source code for Cylinder gravity contribution. (Blakely, 1996)
# --------------------------------------------------------------------------------------------------

import numpy as np

def cylinder(xo, yo, zo, model):
    '''
    This function is a Python implementation which computes the vertical gravitational attraction due 
    to a cylinder. The reference is on Blakely (1996).
    '''
    
    # Setting some constants
    km2m = 0.001
    G = 6.673e-11
    si2mGal = 100000.0
    
    xc, zc = model[0], model[1]
    a, rho = model[2], model[3]
      
    # Calculates the distances
    x = xo - xc
    y = yo - xc
    z = zo - zc
    # Computing r square
    r2 = x**2 + y**2 + z**2
    
    # Calculate the constant
    mass = (np.pi) * (a**2) * rho
    
    # Computes gx and gz
    gx = (-2. * mass * x)/r2
    gz = (-2. * mass * z)/r2
    
    # Conversion to mGal
    gx *= G*si2mGal*km2m
    gz *= G*si2mGal*km2m
    
    # Return the tinal output
    return gx, gz
