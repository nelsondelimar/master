# -----------------------------------------------------------------------------------
# Title: Auxiliars
# Author: Nelson Ribeiro Filho
# Description: Auxiliars codes for using Gravmag Codes
# Collaborator: Rodrigo Bijani
# -----------------------------------------------------------------------------------

import warnings
import numpy as np
from scipy.interpolate import griddata

def padzeros(vector, width, ax, kwargs):
    '''
    This function pads an array with zeros. It should be used while converting or expanding a 
    simple 1D array or a 2D grid, along the pad function which belongs to numpy packages.
    
    Inputs: 
    vector - numpy array - input data
    width - integer - number of values padded
    iaxis - integer - axis which will be calculated
    kwargs - string - keywords arguments
    
    Output:
    newvec - numpy array - the new value for the vector
    '''

    # Padding with zeros on both axis
    vector[:width[0]] = 0.
    vector[-width[1]:] = 0.
    
    # Return the final output
    return vector

def padones(vector, width, ax, kwargs):
    '''
    This function is similar to padzeros functions, but it adds the one value on the axis instead 
    of zeros. It has the same inputs and outputs.
    '''
    
    # Padding with zeros on both axis
    vector[:width[0]] = 1.
    vector[-width[1]:] = 1.
    
    # Return the final output
    return vector

def ispower2(data):
    '''
    Logical function that states if the data has size or shape in a power of two.
    If data is a 1D array, it returns True or False. Otherwise if data is a 2D
    array, it returns a set with (True/False, True/False).
    
    Input: 
    data - numpy array - vector or matrix
    
    Output:
    base - float - number in power of two
    '''
    
    # Verify if data is a vector or a matrix
    if data.size == data.shape[0]:
        num = data.size
        a = num/2.
        b = num//2.
        return np.allclose(a,b)
    elif data.size != data.shape[0]:
        # To shape(0) calculation
        num0 = data.shape[0]
        a, b = num0/2., num0//2.
        # To shape(1) calculation
        num1 = data.shape[1]
        c, d = num1/2., num1//2.
        return np.allclose(a,b), np.allclose(c,d)
    
def nextpower2(num):
    '''
    This function is called in the paddata function to stablish which is
    the next power of two if the grid has no power two in the data set.    
    
    Input:
    num - integer - size of the data shape
    Output:
    next - float - next number power of two    
    
    Ps. This function was modified from Fatiando A Terra.
    '''
    
    # Asserting some conditions
    assert num != 0., 'Data shape should be different of zero!'
    assert num > 0., 'Shape must have no negative values!'
    
    # Calculating the next power of 2
    next = np.ceil(np.log(num)/np.log(2))
    # Return the final output
    return int(2**next)


def paddata(data, shape):
    n = nextpower2(np.max(shape))
    nx, ny = shape
    padx = (n - nx)//2
    pady = (n - ny)//2
    padded = np.pad(data.reshape(shape),
                    ((padx, padx), 
                     (pady, pady)),
                    mode='edge')
    return padded
    
def gridding(x, y, values, datashape):
    '''
    This function creates a grid for the data input and interpolate the values using a 
    low extrapolate and cubic method. It receives 3 1D array vector (x, y and z) and
    also the shape
    
    Input:
    x - numpy array - elements for x direction
    y - numpy array - elements for y direction
    zvalues - numpy array - values that will be interpolate
    datashape - float - shape of the data to interpolate
    '''

    # Setting the area of the data
    area = [x.min(), x.max(), y.min(), y.max()]
    # Initial and final values for x and y
    xi, xf, yi, yf = area
    # Number of points in x and y directions
    nx, ny = datashape
    
    # Creating the vectors for x and y
    xn = np.linspace(xi, xf, nx)
    yn = np.linspace(yi, yf, ny)
    # Creating the grid
    xp, yp = np.meshgrid(xn, yn)[::-1]
    # Eliminatign the values on the edges
    xp = xp.ravel()
    yp = yp.ravel()

    # Allowing the extrapolate of the data
    # Setting the algorithm
    extrapolate = True
    algorithm = 'cubic' 
    
    # Calling the Scipy function
    grid = griddata((x, y), values, (xp, yp), method = 'cubic').ravel()

    # Allowing the extrapolate and also the nearest gridding
    if extrapolate and algorithm != 'nearest' and np.any(np.isnan(grid)):
        if np.ma.is_masked(grid):
            nans = grid.mask
        else:
            nans = np.isnan(grid)
            notnans = np.logical_not(nans)
            grid[nans] = griddata((x[notnans], y[notnans]), grid[notnans],
                                  (x[nans], y[nans]),
                                  method='nearest').ravel()
    
    # Setting the values for x, y and z as the same shape
    xp = xp.reshape(shape)
    yp = yp.reshape(shape)
    grid = grid.reshape(shape)
    
    # Return the final output
    return (xp, yp, grid)
