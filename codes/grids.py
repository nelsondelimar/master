# -----------------------------------------------------------------------------------
# Title: Grids
# Author: Nelson Ribeiro Filho
# Description: Source codes for grid creation and manipulation
# Collaborator: Rodrigo Bijani
# -----------------------------------------------------------------------------------

import numpy
import scipy
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
    np.random.seed(seed)
    
    # Define the arrays
    xarray = numpy.random.uniform(x1, x2, n)
    yarray = numpy.random.uniform(y1, y2, n)
    # If Z is not give:
    if z is not None:
        zarray = z*nump.ones(n)
    return xarray, yarray, zarray
    
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
    # Creating the grid
    xp, yp = regular_grid(area, datashape)[::-1]
    # Eliminatign the values on the edges
    xp = xp.ravel()
    yp = yp.ravel()
    # Allowing the extrapolate of the data
    # Setting the algorithm
    extrapolate = True
    algorithm = 'cubic' 
    # Calling the Scipy function
    grid = scipy.interpolate.griddata((x, y), values, (xp, yp), method = 'cubic').ravel()
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
    data = grid.reshape(shape)
    # Return the final output
    return xp, yp, data

def circle_points(area, n, z=None, random=False, seed=None):

    x1, x2, y1, y2 = area
    radius = 0.5 * min(x2 - x1, y2 - y1)
    if random:
        numpy.random.seed(seed)
        angles = numpy.random.uniform(0, 2 * math.pi, n)
        numpy.random.seed()
    else:
        da = 2. * math.pi / float(n)
        angles = numpy.arange(0., 2. * math.pi, da)
    xs = 0.5 * (x1 + x2) + radius * numpy.cos(angles)
    ys = 0.5 * (y1 + y2) + radius * numpy.sin(angles)
    # Calculate z
    if z is not None:
        zs = z*numpy.ones(n)
    return xy, ys, zs

def cut_grid(x, y, scalars, area):
    xmin, xmax, ymin, ymax = area
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    inside = [i for i in xrange(len(x))
              if x[i] >= xmin and x[i] <= xmax
              and y[i] >= ymin and y[i] <= ymax]
    return [x[inside], y[inside], [s[inside] for s in scalars]]

def loadGRD_surfer(fname):
    
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
    return x.reshape(shape), y.reshape(shape), data.reshape(shape)