# --------------------------------------------------------------------------------------------------
# Title: Grav-Mag Codes
# Author: Nelson Ribeiro Filho
# Description: Source codes that will be necessary during the masters course.
# Collaborator: Rodrigo Bijani
# --------------------------------------------------------------------------------------------------

# Import Python libraries
import numpy as np
# Import my libraries
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
                tfa += ((-1.)**(i + j))*mag*(0.5*(MF[2])*np.log((R - A[i])/(R + A[i])) + 0.5*(MF[1])*
                                             np.log((R - B[j])/(R + B[j])) - (MF[0])*np.log(R + H[k]) -
                                             (MF[3])*np.arctan2(AxB, X2 + HxR + H2) -
                                             (MF[4])*np.arctan2(AxB, R2 + HxR - X2) +
                                             (MF[5])*np.arctan2(AxB, HxR))
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
       
    # Definitions for all distances
    x = [prism[1] - xo, prism[0] - xo]
    y = [prism[3] - yo, prism[2] - yo]
    z = [prism[5] - zo, prism[4] - zo]
    
    # Definition for density
    rho = prism[6]
    
    # Definition - some constants
    G = 6.673e-11
    si2mGal = 100000.0
    
    # Creating the zeros vector to allocate the result
    potential = np.zeros_like(xo)
    
    # Solving the integral as a numerical approximation
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = np.sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                result = (x[i]*y[j]*np.log(z[k] + r)
                          + y[j]*z[k]*np.log(x[i] + r)
                          + x[i]*z[k]*np.log(y[j] + r)
                          - 0.5*x[i]**2 *
                          np.arctan2(z[k]*y[j], x[i]*r)
                          - 0.5*y[j]**2 *
                          np.arctan2(z[k]*x[i], y[j]*r)
                          - 0.5*z[k]**2*np.arctan2(x[i]*y[j], z[k]*r))
                potential += ((-1.)**(i + j + k))*result*rho
    
    # Multiplying the values for 
    potential *= G
        
    # Return the final output
    return potential

def prism_gx(xo, yo, zo, prism):
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
    
    # Definitions for all distances
    x = [prism[1] - xo, prism[0] - xo]
    y = [prism[3] - yo, prism[2] - yo]
    z = [prism[5] - zo, prism[4] - zo]
    
    # Definition for density
    rho = prism[6]
    
    # Definition - some constants
    G = 6.673e-11
    si2mGal = 100000.0
    
    # Numpy zeros array to update the result
    gx = np.zeros_like(xo)
    
    # Compute the value for Gz
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = np.sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                result = -(y[j]*np.log(z[k] + r) + z[k]*np.log(y[j] + r) - x[i]*np.arctan2(z[k]*y[j], x[i]*r))
                gx += ((-1.)**(i + j + k))*result*rho

                # Multiplication for all constants and conversion to mGal
    gx *= G*si2mGal
    
    # Return the final output
    return gx

def prism_gy(xo, yo, zo, prism):
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
    
    # Definitions for all distances
    x = [prism[1] - xo, prism[0] - xo]
    y = [prism[3] - yo, prism[2] - yo]
    z = [prism[5] - zo, prism[4] - zo]
    
    # Definition for density
    rho = prism[6]
    
    # Definition - some constants
    G = 6.673e-11
    si2mGal = 100000.0
    
    # Numpy zeros array to update the result
    gy = np.zeros_like(xo)
    
    # Compute the value for Gz
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = np.sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                result = -(z[k]*np.log(x[i] + r) + x[i]*np.log(z[k] + r) - y[j]*np.arctan2(x[i]*z[k], y[j]*r))
                gy += ((-1.)**(i + j + k))*result*rho
                
    # Multiplication for all constants and conversion to mGal
    gy *= G*si2mGal
    
    # Return the final output
    return gy

def prism_gz(xo, yo, zo, prism):
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
    
    # Definitions for all distances
    x = [prism[1] - xo, prism[0] - xo]
    y = [prism[3] - yo, prism[2] - yo]
    z = [prism[5] - zo, prism[4] - zo]
    
    # Definition for density
    rho = prism[6]
    
    # Definition - some constants
    G = 6.673e-11
    si2mGal = 100000.0
    
    # Numpy zeros array to update the result
    gz = np.zeros_like(xo)
    
    # Compute the value for Gz
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = np.sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                result = -(x[i]*np.log(y[j] + r) + y[j]*np.log(x[i] + r) - z[k]*np.arctan2(x[i]*y[j], z[k]*r))
                gz += ((-1.)**(i + j + k))*result*rho
                
    # Multiplication for all constants and conversion to mGal
    gz *= G*si2mGal
    
    # Return the final output
    return gz
