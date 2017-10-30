# ------------------------------------------------------------------------------------------------------------------------------

# Title: Nielsen Codes
# Author: Nelson Ribeiro Filho
# Description: Developed codes that will be necessary during the masters course and it also be useful during the masters project performance.
# Collaborator: Rodrigo Bijani

# ------------------------------------------------------------------------------------------------------------------------------

# Some libraries that we will use on this python file!

import numpy as np
#import pandas as pd
#import random as rdm
#import fatiando as ft
#import matplotlib.pyplot as plt

def degrees_rad(angle):
    
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

def rad_degrees(argument):
    
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

def dir_cossine(theta_inc, theta_dec, theta_azi):
    
    '''
    This function calculates the cossines projected values on directions using inclination and declination values. Here, we do not considerate an azimuth as a zero value, but as a input value.
    
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
    incl = degrees_rad(theta_inc)
    decl = degrees_rad(theta_dec)
    azim = degrees_rad(theta_azi)
    
    # Calculates the projected cossine values
    dirA = np.cos(incl)*np.cos(decl - azim)
    dirB = np.cos(incl)*np.sin(decl - azim)
    dirC = np.sin(incl)
    
    # Return the final output
    return dirA, dirB, dirC

def regional_components(valF, incF, decF, aziF):
    
    '''
    This fucntion computes the projected components of the regional magnetic field in all directions X, Y and Z. This calculation is done by using a cossine projected function, which recieves the values for an inclination, declination and also and azimuth value. It returns all three components for a magnetic field (Fx, Fy e Fz), using a value for the regional field (F) as a reference for the calculation.
    
    Inputs: 
    valF - float - regional magnetic field value
    incF - float - magnetic field inclination value
    decF - float - magnetic field declination value
    aziF - float - magnetic field azimuth value
    
    Outputs:
    vecF - numpy array - F componentes along X, Y e Z axis
        
    Ps. All inputs can be arrays when they are used for a set of values.
    
    '''
    
    assert valF != 0., 'Value of the regional magnetic field must be nonzero!'
        
    # Computes the projected cossine
    X, Y, Z = dir_cossine(incF, decF, aziF)
    # Compute all components
    Fx = valF*X
    Fy = valF*Y
    Fz = valF*Z
    # Set the F values as an array
    vecF = np.array([Fx, Fy, Fz])
    # Return the final output
    return vecF

def dipole(x, y, z, xe, ye, ze, raio, mag, inc, dec, az):
    
    '''
    
    This function is a Python implementation for a Fortran subroutine contained in Blakely (1995). It computes the components of the magnetic induction (Bx, By, Bz) caused by a sphere with uniform distribution of magnetization. The direction X represents the north and Z represents growth downward. This function receives the coordinates of the points of observation (X, Y, Z - arrays), the coordinates of the center of the sphere (Xe, Ye, Ze), the magnetization intensity M and the values for inclination and declination (in degrees). The observation values are given in meters.
    
    Inputs: 
    x, y, z - numpy arrays - position of the observation points
    xe, ye, ze - array - position of the center of the sphere
    mag - flaot - magnetization intensity value
    inc - float - inclination value
    dec - float - declination value
    
    Outputs:
    Bx - componente do campo magnetico induzido na direcao X
    By - componente do campo magnetico induzido na direcao Y
    Bz - componente do campo magnetico induzido na direcao Z
    
    Ps. The value for Z can be a scalar in the case of one depth, otherwise it can be a set of points.
    
    '''
    #assert x.size == y.size 
    #assert x.shape[0] == x.shape[1]
    
    # Calculates some constants
    t2nt = 1.e9 # Testa to nT - conversion
    cm = 1.e-7  # Magnetization constant
    
    # Distances in all axis directions - x, y e z
    rx = x - xe
    ry = y - ye
    rz = z - ze
    # Computes the distance (r) as the module of the other three components
    r = np.sqrt((rx**2)+(ry**2)+(rz**2))
    # Computes the magnetization values for all directions
    mx, my, mz = dir_cossine(inc, dec, az)
    
    # Auxiliars calculations
    dot = (rx*mx) + (ry*my) + (rz*mz)  # Scalar product
    m = (4.*np.pi*(raio**3)*mag)/3.    # Magnetic moment
    
    # Components calculation - Bx, By e Bz
    Bx = cm*m*(3.*dot*rx-(r**2*mx))/r**5
    By = cm*m*(3.*dot*ry-(r**2*my))/r**5
    Bz = cm*m*(3.*dot*rz-(r**2*mz))/r**5
    # Final components calculation
    Bx *= t2nt
    By *= t2nt
    Bz *= t2nt
    # Return the final output
    return Bx, By, Bz

def grav_sphere(xobs, yobs, sphere):
    '''
    
    This function calculates the gravity contribution due to a solid sphere. This is a Python implementation for the subroutine presented in Blakely (1995).
    On this function, there are received the value of the initial and final observation points and the properties of the sphere. The inputs sphere is allocated as:
    sphere[size = 5] = sphere[x center, y center, z center, radius , density]
    
    Inputs:
    sphere - numpy array - elements of the sphere
    
    Output:
    gz - numpy array - vertical component for the gravity signal due to a solid sphere
    '''
    
    #assert xobs[0] >= sphere[0], 'Grid value in X direction must have greater than the X position for the sphere'
    #assert xobs[0] >= sphere[1], 'Grid value in X direction must have greater than the Y position for the sphere'
    #assert yobs[1] >= sphere[0], 'Grid value in Y direction must have greater than the X position for the sphere'
    #assert yobs[1] >= sphere[1], 'Grid value in Y direction must have greater than the Y position for the sphere'
    
    # Definition for some constants
    gamma = 6.67e-6
    dmy = 1e-5 # Dummy value
    
    # Compute the constant which is result due to the product
    C = (4./3)*np.pi*gamma*sphere[4]*(sphere[3]**3)
    
    gz_signal = C*sphere[2]/(((xobs + dmy - sphere[0])**2 + (yobs + dmy - sphere[1])**2 + (sphere[2]**2))**(3./2))
    
    return gz_signal