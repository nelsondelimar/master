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

def irregular_grid(area, n, z = None, seed = None):
    '''
    This function creates a rregular grid, once the area, the shape and the level are given as input. 
    It also asserts that area must have four elements and the final values must greater than initial values. The
    level indicates the value over the grid, which is converted for an array with same shape of x and y.
    
    Inputs:
    area - numpy list - initial and final values
    n - integer - number of points
    level - float - level of observation (positive downward)
    
    Outputs:
    xp, yp - numpy 2D array - grid of points
    zp - numpy 2D array - gird at observation level
    '''

    xi, xf, yi, yf = area
    
    # Condition
    if xi > xf or yi > yf:
        raise ValueError('Final values must be greater than initial values!')

    # Including the seed
    numpy.random.seed(seed)
    
    # Define the arrays
    xarray = numpy.random.uniform(xi, xf, n)
    yarray = numpy.random.uniform(yi, yf, n)
    # If Z is not give:
    if z is not None:
        zarray = z*numpy.ones(n)
    return xarray, yarray, zarray

def profile(x, y, data, p1, p2, size):
    ''' It draws a interpolated profile between two data points. It recieves the original 
    observation points and data, and returns the profile.
    
    Inputs:
    x, y - numpy array - observation points
    data - numpy array - observed data
    p1 - list - initial profile point (x,y)
    p2 - list - final profile point (x,y)
    size - scalar - number of points along profile
    
    Output:
    profile - numpy array - interpolated profile
    '''
    
    # Defines points coordinates
    x1, y1 = p1
    x2, y2 = p2
    
    # Calculate the distances
    maxdist = numpy.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    distances = numpy.linspace(0, maxdist, size)
    
    # Angle of profile line
    angle = numpy.arctan2(y2 - y1, x2 - x1)
    xp = x1 + distances * numpy.cos(angle)
    yp = y1 + distances * numpy.sin(angle)    
    
    # Calculates the interpolated profile    
    profile = griddata((x, y), data, (xp, yp), method = 'cubic')
    # Return the final output
    return xp, yp, profile
    
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

def load_surfer(fname):
    """
    Read a Surfer grid file and return three 1d numpy arrays and the grid shape

    Surfer is a contouring, gridding and surface mapping software
    from GoldenSoftware. The names and logos for Surfer and Golden
    Software are registered trademarks of Golden Software, Inc.

    http://www.goldensoftware.com/products/surfer

    Parameters:

    * fname : str
        Name of the Surfer grid file
    * fmt : str
        File type, can be 'ascii' or 'binary'

    Returns:

    * x : 1d-array
        Value of the North-South coordinate of each grid point.
    * y : 1d-array
        Value of the East-West coordinate of each grid point.
    * data : 1d-array
        Values of the field in each grid point. Field can be for example
        topography, gravity anomaly etc
    * shape : tuple = (nx, ny)
        The number of points in the x and y grid dimensions, respectively

    """
    with open(fname) as ftext:
        # DSAA is a Surfer ASCII GRD ID
        id = ftext.readline()
        # Read the number of columns (ny) and rows (nx)
        ny, nx = [int(s) for s in ftext.readline().split()]
        shape = (nx, ny)
        # Read the min/max value of columns/longitude (y direction)
        ymin, ymax = [float(s) for s in ftext.readline().split()]
        # Read the min/max value of rows/latitude (x direction)
        xmin, xmax = [float(s) for s in ftext.readline().split()]
        area = (xmin, xmax, ymin, ymax)
        # Read the min/max value of grid values
        datamin, datamax = [float(s) for s in ftext.readline().split()]
        data = numpy.fromiter((float(i) for line in ftext for i in
                               line.split()), dtype='f')
        data = numpy.ma.masked_greater_equal(data, 1.70141e+38)
        assert numpy.allclose(datamin, data.min()) \
        and numpy.allclose(datamax, data.max()), \
        "Min and max values of grid don't match ones read from file." \
        + "Read: ({}, {})  Actual: ({}, {})".format(
            datamin, datamax, data.min(), data.max())
        # Create x and y coordinate numpy arrays
        x, y = regular_grid(area, shape)
    return x, y, data, shape