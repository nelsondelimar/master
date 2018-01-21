# --------------------------------------------------------------------------------------------------
# Title: Grav-Mag Codes
# Author: Nelson Ribeiro Filho
# Description: Source codes that will be necessary during the masters course.
# Collaborator: Rodrigo Bijani
# --------------------------------------------------------------------------------------------------

import numpy as np
import auxiliars as aux

def prism_tf(x, y, z, prism, directions, field):
    '''
    This function calculates the total field anomaly produced by a rectangular prism located under 
    surface; it is a Python implementation for the Subroutin MBox which is contained on Blakely (1995). 
    It recieves: the coordinates of the positions in all directions, the elements of the prims, the 
    angle directions and the elements of the field. That function also uses the auxilary function 
    DIR_COSSINE to calculate the projections due to the field F and the source S.
    
    Inputs:
    x, y - numpy arrays - observation points in x and y directions
    z - numpy array/float - height for the observation
    prism - numpy array - all elements for the prims
        prism[0, 1] - initial and final coordinates at X (dimension at X axis!)
        prism[2, 3] - initial and final coordinates at Y (dimension at Y axis!)
        prism[4, 5] - initial and final coordinates at Z (dimension at Z axis!)
        prism[6] - magnetic intensity
    directions - numpy array - elements for source directions
        directions[0] - float - source inclination
        directions[1] - float - source declination
    field - numpy array - elementes for regional field
        field[0] - float - magnetic field inclination
        field[1] - float - magnetic field declination
        
    Output:
    tfa - numpy array - calculated total field anomaly
    
    X and Y represents North and East; Z is positive downward.
    Ps. Z can be a array with all elements for toppography or a float point as a flight height.
    '''    
    
    # Stablishing some conditions
    if x.shape != y.shape:
        raise ValueError("All inputs must have same shape!")
        
    # Stablish some constants
    t2nt = 1.e9 # Testa to nT - conversion
    cm = 1.e-7  # Magnetization constant

    # Setting some values
    D1 = directions[0]
    D2 = directions[1]
    F1 = field[0]
    F2 = field[1] 
    
    # Calculate the directions for the source magnetization and for the field
    Ma, Mb, Mc = aux.dircos(D1, D2) # s -> source
    Fa, Fb, Fc = aux.dircos(F1, F2) # f -> field

    # Aranges all values as a vector
    MF = [Ma*Fb + Mb*Fa, 
          Ma*Fc + Mc*Fa, 
          Mb*Fc + Mc*Fb, 
          Ma*Fa, 
          Mb*Fb, 
          Mc*Fc]
    
    # Limits for initial and final position along the directions
    A = [prism[1] - x, prism[0] - x]
    B = [prism[3] - y, prism[2] - y]
    H = [prism[5] - z, prism[4] - z]
    
    # Create the zero array to allocate the total field result
    tfa = np.zeros_like(x)
    
    # Loop for controling the signal of the function    
    mag = prism[6]
    for k in range(2):
        mag *= -1
        H2 = H[k]**2
        for j in range(2):
            Y2 = B[j]**2
            for i in range(2):
                X2 = A[i]**2
                AxB = A[i]*B[j]
                R2 = X2 + Y2 + H2
                R = np.sqrt(R2)
                HxR = H[k]*R
                tfa += ((-1.)**(i + j))*mag*(0.5*(MF[2])*aux.my_log((R - A[i])/(R + A[i])) + 0.5*(MF[1])*
                                             aux.my_log((R - B[j])/(R + B[j])) - (MF[0])*aux.my_log(R + H[k]) -
                                             (MF[3])*aux.my_atan(AxB, X2 + HxR + H2) -
                                             (MF[4])*aux.my_atan(AxB, R2 + HxR - X2) +
                                             (MF[5])*aux.my_atan(AxB, HxR))
    # Multiplying for constants conversion
    tfa *= t2nt*cm
    
    # Return the final output
    return tfa

def potential(xo, yo, zo, prism):
    '''
    This function calculates the gravitational potential due to a rectangular prism. It is calculated 
    solving a numerical integral approximated by using the gravity field G(x,y,z), once G can be written 
    as minus the gradient of the gravitational potential. This function recieves all obsevation points 
    for an array or a grid and also the value for height of the observation, which can be a simple float 
    number (as a level value) or a 1D array. It recieves the values for the prism dimension in X, Y and Z 
    directions.
    
    Inputs:
    x, y - numpy arrays - observation points in x and y directions
    z - numpy array/float - height for the observation
    prism - numpy array - all elements for the prims
        prism[0, 1] - initial and final coordinates at X (dimension at X axis!)
        prism[2, 3] - initial and final coordinates at Y (dimension at Y axis!)
        prism[4, 5] - initial and final coordinates at Z (dimension at Z axis!)
        prism[6] - density value
        
    Output:
    potential - numpy array - gravitational potential due to a solid prism
    '''

    # Stablishing some conditions
    if x.shape != y.shape:
        raise ValueError("All inputs must have same shape!")
       
    # Definitions for all distances
    xp = [prism[1] - x, prism[0] - x]
    yp = [prism[3] - y, prism[2] - y]
    zp = [prism[5] - z, prism[4] - z]
    
    # Definition for density
    rho = prism[6]
    
    # Definition - some constants
    G = 6.673e-11
    si2mGal = 100000.0
    
    # Creating the zeros vector to allocate the result
    potential = np.zeros_like(x)
    
    # Solving the integral as a numerical approximation
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = np.sqrt(xp[i]**2 + yp[j]**2 + zp[k]**2)
                result = (xp[i]*yp[j]*aux.my_log(zp[k] + r)
                          + yp[j]*zp[k]*aux.my_log(xp[i] + r)
                          + xp[i]*zp[k]*aux.my_log(yp[j] + r)
                          - 0.5*xp[i]**2 *
                          aux.my_atan(zp[k]*yp[j], xp[i]*r)
                          - 0.5*yp[j]**2 *
                          aux.my_atan(zp[k]*xp[i], yp[j]*r)
                          - 0.5*zp[k]**2*aux.my_atan(xp[i]*yp[j], zp[k]*r))
                potential += ((-1.)**(i + j + k))*result*rho
    
    # Multiplying the values for 
    potential *= G
        
    # Return the final output
    return potential

def prism_gx(x, y, z, prism):
    '''
    This function is a Python implementation for the X horizontal component for the gravity field due to 
    a rectangular prism, which has initial and final positions equals to xi and xf, yi and yf, for the X 
    and Y directions. This function also recieve the obsevation points for an array or a grid and also the 
    value for height of the observation, which can be a simple float number (as a level value) or a 1D array.
    
    Inputs:
    x, y - numpy arrays - observation points in x and y directions
    z - numpy array/float - height for the observation
    prism - numpy array - all elements for the prims
        prism[0, 1] - initial and final coordinates at X (dimension at X axis!)
        prism[2, 3] - initial and final coordinates at Y (dimension at Y axis!)
        prism[4, 5] - initial and final coordinates at Z (dimension at Z axis!)
        prism[6] - density value
        
    Output:
    gx - numpy array - vertical component for the gravity atraction
    '''

    # Stablishing some conditions
    if x.shape != y.shape:
        raise ValueError("All inputs must have same shape!")
       
    # Definitions for all distances
    xp = [prism[1] - x, prism[0] - x]
    yp = [prism[3] - y, prism[2] - y]
    zp = [prism[5] - z, prism[4] - z]    

    # Definition for density
    rho = prism[6]
    
    # Definition - some constants
    G = 6.673e-11
    si2mGal = 100000.0
    
    # Numpy zeros array to update the result
    gx = np.zeros_like(x)
    
    # Compute the value for Gx
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = np.sqrt(xp[i]**2 + yp[j]**2 + zp[k]**2)
                result = -(yp[j]*aux.my_log(zp[k] + r) + zp[k]*aux.my_log(yp[j] + r) - xp[i]*aux.my_atan(zp[k]*yp[j], xp[i]*r))
                gx += ((-1.)**(i + j + k))*result*rho

                # Multiplication for all constants and conversion to mGal
    gx *= G*si2mGal
    
    # Return the final output
    return gx

def prism_gy(x, y, z, prism):
    '''
    This function is a Python implementation for the Y horizontal  component for the gravity field due 
    to a rectangular prism, which has initial and final positions equals to xi and xf, yi and yf, for 
    the X and Y directions. This function also recieve the obsevation points for an array or a grid and 
    also the value for height of the observation, which can be a simple float number (as a level value) 
    or a 1D array.
    
    Inputs:
    x, y - numpy arrays - observation points in x and y directions
    z - numpy array/float - height for the observation
    prism - numpy array - all elements for the prims
        prism[0, 1] - initial and final coordinates at X (dimension at X axis!)
        prism[2, 3] - initial and final coordinates at Y (dimension at Y axis!)
        prism[4, 5] - initial and final coordinates at Z (dimension at Z axis!)
        prism[6] - density value
        
    Output:
    gy - numpy array - vertical component for the gravity atraction
    '''
    
    # Stablishing some conditions
    if x.shape != y.shape:
        raise ValueError("All inputs must have same shape!")
       
    # Definitions for all distances
    xp = [prism[1] - x, prism[0] - x]
    yp = [prism[3] - y, prism[2] - y]
    zp = [prism[5] - z, prism[4] - z]  
    
    # Definition for density
    rho = prism[6]
    
    # Definition - some constants
    G = 6.673e-11
    si2mGal = 100000.0
    
    # Numpy zeros array to update the result
    gy = np.zeros_like(x)
    
    # Compute the value for Gy
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = np.sqrt(xp[i]**2 + yp[j]**2 + zp[k]**2)
                result = -(zp[k]*aux.my_log(xp[i] + r) + xp[i]*aux.my_log(zp[k] + r) - yp[j]*aux.my_atan(xp[i]*zp[k], yp[j]*r))
                gy += ((-1.)**(i + j + k))*result*rho
                
    # Multiplication for all constants and conversion to mGal
    gy *= G*si2mGal
    
    # Return the final output
    return gy

def prism_gz(x, y, z, prism):
    '''
    This function is a Python implementation for the vertical component for the gravity field due to a 
    rectangular prism, which has initial and final positions equals to xi and xf, yi and yf, for the X 
    and Y directions. This function also recieve the obsevation points for an array or a grid and also 
    the value for height of the observation, which can be a simple float number (as a level value) or a 
    1D array.
    
    Inputs:
    x, y - numpy arrays - observation points in x and y directions
    z - numpy array/float - height for the observation
    prism - numpy array - all elements for the prims
        prism[0, 1] - initial and final coordinates at X (dimension at X axis!)
        prism[2, 3] - initial and final coordinates at Y (dimension at Y axis!)
        prism[4, 5] - initial and final coordinates at Z (dimension at Z axis!)
        prism[6] - density value
        
    Output:
    gz - numpy array - vertical component for the gravity atraction
    '''

    # Stablishing some conditions
    if x.shape != y.shape:
        raise ValueError("All inputs must have same shape!")
       
    # Definitions for all distances
    xp = [prism[1] - x, prism[0] - x]
    yp = [prism[3] - y, prism[2] - y]
    zp = [prism[5] - z, prism[4] - z]  
    
    # Definition for density
    rho = prism[6]
    
    # Definition - some constants
    G = 6.673e-11
    si2mGal = 100000.0
    
    # Numpy zeros array to update the result
    gz = np.zeros_like(x)
    
    # Compute the value for Gz
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = np.sqrt(xp[i]**2 + yp[j]**2 + zp[k]**2)
                result = -(xp[i]*aux.my_log(yp[j] + r) + yp[j]*aux.my_log(xp[i] + r) - zp[k]*aux.my_atan(xp[i]*yp[j], zp[k]*r))
                gz += ((-1.)**(i + j + k))*result*rho
                
    # Multiplication for all constants and conversion to mGal
    gz *= G*si2mGal
    
    # Return the final output
    return gz