# -----------------------------------------------------------------------------------
# Title: Auxiliars
# Author: Nelson Ribeiro Filho
# Description: Auxiliars codes for using Gravmag Codes
# Collaborator: Rodrigo Bijani
# -----------------------------------------------------------------------------------

import numpy as np # Numpy library
import scipy as sp # Scipy library
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

def dircos(inc, dec):
    
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
        
    # Calculates the projected cossine values
    A = np.cos(incl)*np.cos(decl)
    B = np.cos(incl)*np.sin(decl)
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

def addnoise(data, std):
    
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
    local = np.abs((data.max() - data.min())*(1e-2))
    
    # Verify if data is a 1D or 2D array
    if data.shape[0] == size or data.shape[1] == size:
        noise += np.random.normal(loc = local, scale = std, size = size)
    else:
        noise += np.random.normal(loc = local, scale = std, size = shape)
        
    # Return the final output
    return data + 2*noise - 0.5*noise.mean()

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
    
def datagrid(x, y, zvalues, datashape):

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
    grid = scipy.interpolate.griddata((x, y), zvalues, (xp, yp), method = 'cubic').ravel()

    # Allowing the extrapolate and also the nearest gridding
    if extrapolate and algorithm != 'nearest' and np.any(np.isnan(grid)):
        if np.ma.is_masked(grid):
            nans = grid.mask
        else:
            nans = np.isnan(grid)
            notnans = np.logical_not(nans)
            grid[nans] = scipy.interpolate.griddata((x[notnans], y[notnans]), grid[notnans],
                                         (x[nans], y[nans]),
                                         method='nearest').ravel()
    
    # Setting the values for x, y and z as the same shape
    xp = xp.reshape(shape)
    yp = yp.reshape(shape)
    grid = grid.reshape(shape)
    
    # Return the final output
    return (xp, yp, grid)