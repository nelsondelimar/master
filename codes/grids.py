# -----------------------------------------------------------------------------------
# Title: Grids
# Author: Nelson Ribeiro Filho
# Description: Source codes for grid creation and manipulation
# Collaborator: Rodrigo Bijani
# -----------------------------------------------------------------------------------

import numpy
import warnings
from scipy.interpolate import griddata

def regular_grid(area, shape, level = None):
    '''
    This function creates a regular grid, once the area, the shape and the level are given as input. 
    The area must have four elements named as [xi, xf, yi, yf].THe shape represents the grid size. The
    level indicates the value over the grid, which is converted for an array with same shape of x and y.
    
    Inputs:
    area - numpy list - initial and final values
    shape - tuple - number of elements in x and y
    level - float - level of observation (positive downward)
    
    Outputs:
    xp, yp - numpy 2D array - grid of points
    zp - numpy 2D array - gird at observation level    
    '''
    
    # Defines the initial and final values for grid creation
    xi, xf, yi, yf = area
    
    # Condition
    if xi > xf or yi > yf:
        raise ValueError('Final values must be greater than initial values!')
        
    # Number of elements on the grid
    nx, ny = shape
    
    # Creates the vectors in x and y directions
    x = numpy.linspace(xi, xf, nx)
    y = numpy.linspace(yi, yf, ny)
   
    # Grid in that order, once meshgrid uses the first argument as columns
    yp, xp = numpy.meshgrid(y, x)
    # Condition fot the observation level 
    if level is not None:
        zp = level*numpy.ones(nx*ny)
        # Reshape zp - level of observation
        return xp.reshape(nx*ny), yp.reshape(nx*ny), zp
    else:
        # If zp is not given, returns xp and yp only
        return xp.reshape(nx*ny), yp.reshape(nx*ny)

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
    xn = numpy.linspace(xi, xf, nx)
    yn = numpy.linspace(yi, yf, ny)
    # Creating the grid
    xp, yp = numpy.meshgrid(xn, yn)[::-1]
    # Eliminatign the values on the edges
    xp = xp.ravel()
    yp = yp.ravel()

    # Allowing the extrapolate of the data
    # Setting the algorithm
    extrapolate = True
    algorithm = 'cubic' 
    
    # Calling the Scipy function
    grid = scipy.interpolate.griddata((x, y), zvalues, (xp, yp), method = 'cubic').ravel()

    # Allowing the extrapolate and also the nearest gridding
    if extrapolate and algorithm != 'nearest' and numpy.any(numpy.isnan(grid)):
        if numpy.ma.is_masked(grid):
            nans = grid.mask
        else:
            nans = numpy.isnan(grid)
            notnans = numpy.logical_not(nans)
            grid[nans] = scipy.interpolate.griddata((x[notnans], y[notnans]), grid[notnans],
                                         (x[nans], y[nans]),
                                         method='nearest').ravel()
    
    # Setting the values for x, y and z as the same shape
    xp = xp.reshape(shape)
    yp = yp.reshape(shape)
    grid = grid.reshape(shape)
    
    # Return the final output
    return xp, yp, grid
