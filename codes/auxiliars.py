from __future__ import division
import numpy
import scipy
import warnings

def my_deg2rad(angle):
    '''
    It converts an angle value in degrees to radian.     
    
    Input:
    angle - float - number or list of angle in degrees    
    Output:
    argument - float - angle in radian    
    '''
    # Return the final output
    return (angle/180.)*numpy.pi

def my_rad2deg(argument):
    '''
    This function converts an angle value in radian to an another value in degrees.
    
    Input:
    argument - float - number or list of angle in radian
    Output:
    angle - float - angle in degrees    
    '''
    # Return the final output
    return (argument/numpy.pi)*180.

def my_trigonometrics_deg(angle):
    '''
    Return the sine, cosine and tangent functions of an angle or a set of values in degrees.
    
    Input:
    x - numpy float or 1D array - angle in degrees
    
    Output:
    mysin - numpy float or 1D array - sine function
    mycos - numpy float or 1D array - cosine function
    mytan - numpy float or 1D array - tangent function
    '''
    # Conversion factor 
    deg2rad = numpy.pi/180.
    # Calculate sine, cosine and tangent in radian
    mysin = numpy.sin(angle*deg2rad)
    mycos = numpy.cos(angle*deg2rad)
    mytan = numpy.tan(angle*deg2rad)
    # Return the final output
    return mysin, mycos, mytan

def my_trigonometrics_rad(angle):
    '''
    Return the sine, cosine and tangent functions of an angle or a set of values in radians.
    
    Input:
    x - numpy float or 1D array - angle in radians
    
    Output:
    mysin - numpy float or 1D array - sine function
    mycos - numpy float or 1D array - cosine function
    mytan - numpy float or 1D array - tangent function
    '''
    # Conversion factor 
    rad2deg = 180./numpy.pi
    # Calculate sine, cosine and tangent in radian
    mysin = numpy.sin(angle*rad2deg)
    mycos = numpy.cos(angle*rad2deg)
    mytan = numpy.tan(angle*rad2deg)
    # Return the final output
    return mysin, mycos, mytan

def my_asin(x):
    '''
    Return the more stable output for arcsin calculation.
    '''
    # Return the final output
    return numpy.arcsin(x)

def my_acos(x):
    '''
    Return the more stable output for arccs calculation.
    '''
    # Return the final output
    return numpy.arccos(x)

def my_atan(x, y):
    '''
    Return the more stable output for arctan calculation by correcting the 
    value of the angle in arctan2, in order to fix the sign of the tangent.
    '''
    arctan = numpy.arctan2(x, y)
    arctan[x == 0] = 0
    arctan[(x > 0) & (y < 0)] -= numpy.pi
    arctan[(x < 0) & (y < 0)] += numpy.pi
    return arctan

def my_sqrt(x):
    '''
    Return the more stable output for square root calculation by correcting the 
    value if x is negative.
    
    Input:
    x - float or 1D array - input number
    
    Output:
    mysqrt - float or 1D array - square root
    '''
    # Return the final output
    return scipy.sqrt(x)

def my_log(x):
    ''' 
    Return the value 0 for log(0), once the limits applying in the formula
    tend to 0.
    '''
    log = numpy.log(x)
    log[x == 0] = 0
    return log

def my_dot(x,y):
    '''
    Return the safe value for the dot product between two vectors.
    '''
    return numpy.dot(x,y)

def my_hadamard(x,y):
    '''
    Return the safe value for the hadamard product between two vectors.
    '''
    return numpy.multiply(x,y)

def my_outer(x,y):
    '''
    Return the safe value for the outer product between two vectors.
    '''
    return numpy.outer(x,y)

def my_inverse(mat):
    '''
    It returns the safe value for the inverse matrix.
    '''
    return numpy.linalg.inv(mat)

def my_LU(mat):
    '''
    It returns the safe value for the LU decomposition.
    '''
    
    p,l,u = scipy.linalg.lu(mat)
   
    return p,l,u

def my_xrotation(angle):
    '''    
    It returns the rotation matrix given a (x, y, z) point at x direction,
    
    Inputs: 
    angle - numpy float - angle of rotation     
    '''
    #assert angle <= 360, 'Angulo em graus deve ser menor ou igual a 360'
    #assert angle >= 0, 'Angulo em graus deve ser maior ou igual a 0'
    c = numpy.cos((angle/180.)*numpy.pi)
    s = numpy.sin((angle/180.)*numpy.pi)
    # Rotation at x
    rotx = numpy.array([[1., 0., 0.,],
                        [0., c, s],
                        [0., -s, c]])
    # Return the output 
    return rotx

def my_yrotation(angle):
    '''     
    It returns the rotation matrix given a (x, y, z) point at y direction,
    
    Inputs: 
    angle - numpy float - angle of rotation
    Output:
    ry - numpy array 2D - matrix of rotation at y direction    
    ''' 
    #assert angle <= 360, 'Angulo em graus deve ser menor ou igual a 360'
    #assert angle >= 0, 'Angulo em graus deve ser maior ou igual a 0'
    c = numpy.cos((angle/180.)*numpy.pi)
    s = numpy.sin((angle/180.)*numpy.pi)
    # Rotation at y
    roty = numpy.array([[c, 0., s,],
                        [0., 1., 0.],
                        [-s, 0., c]])
    # Return the output
    return roty

def my_zrotation(angle):
    '''    
    It returns the rotation matrix given a (x, y, z) point at z direction,
    
    Inputs: 
    angle - numpy float - angle of rotation
     
    Output:
    rz - numpy array 2D - matrix of rotation at z direction
    '''
     
    #assert angle <= 360, 'Angulo em graus deve ser menor ou igual a 360'
    #assert angle >= 0, 'Angulo em graus deve ser maior ou igual a 0'
    c = numpy.cos((angle/180.)*numpy.pi)
    s = numpy.sin((angle/180.)*numpy.pi)
    # Rotation at z
    rotz = numpy.array([[c, s, 0.,],
                        [-s, c, 0.],
                        [0., 0., 1.]])
    # Return the output
    return rotz

def rotate3D_xyz(x, y, z, angle, direction = 'z'):
    '''   
    It returns the rotated plane x-y along z-axis by default.
    If angle is positive, the rotation in counterclockwise direction; 
    otherwise is clockwise direction.
    
    Inputs:
    x, y, z - numpy arrays - coordinate points
    angle - float - angle of rotation
    direction - string - direction
    
    Outputs:
    xr, yr, zr - numpy arrays - new rotated coordinate points
    '''
    # Matrix rotation in x-y-or-z direction
    if direction == 'x':
        rot = my_xrotation(angle)
    if direction == 'y':
        rot = my_yrotation(angle)
    if direction == 'z':
        rot = my_zrotation(angle)
    # Create the matrix
    mat = numpy.vstack([x, y, z]).T
    # Create the zero matrix
    res = numpy.zeros_like(mat)
    # Calculate the rotated coordinates
    for k, i in enumerate(mat):
        res[k,:] = numpy.dot(rot, i)
    # New coordinates
    xr = res[:,0]
    yr = res[:,1]
    zr = res[:,2]
    # Return the final output
    return xr, yr, zr

def my_spherical2cartesian(longitude, latitude, level):
    '''
    It converts all spherical coordinates values to values in geocentric coordinates. 
    It receives the directions as longitude and latitude and the level.
    
    Inputs:
    longitude - float - longitude value in degrees
    latitude - float - latitude value in degrees
    level - float - height above Earth radius in meters
    
    Outputs:
    x, y, z - floats - converted geocentric coordinates
    '''
    # Defines the value of the Earth radius
    R = 6378137. + level
    
    # Calculates the geocentric coordinates x, y and z
    x = numpy.cos((numpy.pi/180.) * latitude) * numpy.cos((numpy.pi/180.) * longitude) * R
    y = numpy.cos((numpy.pi/180.) * latitude) * numpy.sin((numpy.pi/180.) * longitude) * R
    z = numpy.sin((numpy.pi/180.) * latitude) * R
    
    # Returns the final output
    return x, y, z

def my_normalnoise(xi, vi = 0., std = 0.):
    '''
    It contaminantes the data using a normal Gaussian distribution.
    
    Inputs:
    xi - float or array - input data
    vi - float - standard value for the noise
    std - float - standard deviation
    '''
    # Conditions
    assert numpy.min(xi) <= numpy.mean(xi), 'Mean must be greater than minimum!'
    assert numpy.max(xi) >= numpy.mean(xi), 'Maximum must be greater than mean!'
    assert std >= 0., 'Noise must be greater than zero!'
    # Creat the zero vector such as data
    noise = []#numpy.empty_like(xi)
    # Create the noise:
    for k in range(xi.size):
        noise.append(numpy.random.normal(loc = vi, scale = std))
    # Return the final output
    return xi + numpy.array(noise)

def my_uniformnoise(xi, vmin, vmax):
    '''
    It contaminantes the data using a normal Gaussian distribution.
    
    Inputs:
    xi - float or array - input data
    vmin - float - minimum value
    vmax - float - maximum value
    '''
    # Conditions
    assert numpy.min(xi) <= numpy.mean(xi), 'Mean must be greater than minimum!'
    assert numpy.max(xi) >= numpy.mean(xi), 'Maximum must be greater than mean!'
    # Creat the zero vector such as data
    noise = []#numpy.empty_like(xi)
    # Create the noise:
    for k in range(xi.size):
        noise.append(numpy.random.uniform(vmin, vmax))
    # Return the final output
    return xi + numpy.array(noise)

def my_residual(do, dp):
    '''
    It calculates the residual between the observed data and the calculated predicted data.
    Moreover, perform the calculation for the mean value of this difference, as well as the standard 
    deviation and the normalize value.
    
    Inputs:
    observed - numpy array or list - observed data
    predicted - numpy array or list - predicted data
    
    Outputs:
    res - numpy array or list - difference between observed and predicted
    norm - numpy array or list - norm data values
    mean - float - mean of all values
    std - float - calculated tandard deviation
    '''
    # Condition
    if do.shape != dp.shape:
        raise ValueError("All inputs must have same shape!") 
    # Calculates the residual
    res = do - dp
    # Calculates the mean value of the residual
    mean = numpy.mean(res)
    # Calculates the standard deviation of the residual
    std = numpy.std(res)
    # Calculates the array of norms
    norm = (res - mean)/(std)
    # Returns the output final
    return res, mean, norm, std

def my_dircos(inc, dec, azm = 0.):
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
    Inc = deg2rad(inc)
    Dec = deg2rad(dec)
    Azm = deg2rad(azm)
    # Calculates the projected cossine values
    xdir = numpy.cos(Inc)*numpy.cos(Dec - Azm)
    ydir = numpy.cos(Inc)*numpy.sin(Dec - Azm)
    zdir = numpy.sin(Inc)
    # Return the final output
    return xdir, ydir, zdir

def my_regional(field, inc, dec, azm = 0.):
    '''
    It calculates the regional magnetic field in x, y and z directions.
    It uses values of a reference magnetic field, the inclination and the magnetic declination.
    
    Inputs: 
    field - float - regional magnetic intensity
    inc - float - magnetic field inclination value
    dec - float - magnetic field declination value
    azm - float - magnetic field azimuth value (default = zero)
    
    Outputs:
    fx - float or 1D array - field in x direction
    fy - float or 1D array - field in y direction
    fz - float or 1D array - field in z direction
    '''
    # Computes the projected cossine
    xdir, ydir, zdir = my_dircos(inc, dec, azm)    
    # Compute all components
    fx, fy, fz = field*xdir, field*ydir, field*zdir
    # Set F as array and return the output
    return fx, fy, fz

def my_theta(inc, dec, u, v, azim = 0.):    
    '''
    Return the operators for magnetization and field directions.
    
    Inputs:
    angle - numpy 1D array - inclination and declination
    u - float - number of points in x direction
    v - float - number of points in y direction
    
    Output:
    theta - complex - magnetization projection as a complex number
    '''
    
    # Calculate the modulus for k value. In this case: k = kz
    k = (u**2 + v**2)**(0.5)
    
    # Calcutaing the projections
    x, y, z = my_dircos(inc, dec, azim) 
    theta = z + ((x*u + y*v)/k)*1j
    
    # Return the final output:
    return theta

def my_wavenumber(x, y):
    '''
    Return the wavenumbers in X and Y directions
    
    Inputs:
    x - numpy array - coordinates in x directions
    y - numpy array - coordinates in y directions
    
    Output:
    kx - numpy 2D array - calculated wavenumber in x direction
    ky - numpy 2D array - calculated wavenumber in y direction
    '''
    
    # Verify if x and y are 1D or 2D numpy arrays
    if x.shape[0] == x.size or x.shape[1] == x.size:
        dx = x[1] - x[0]
        nx = x.size
    else:
        dx = (x.max() - x.min())/(x.shape[1] - 1)
        nx = x.shape[1]
    
    if y.shape[0] == y.size or y.shape[1] == y.size:
        dy = y[1] - y[0]
        ny = y.size
    else:
        dy = (y.max() - y.min())/(y.shape[0] - 1)
        ny = y.shape[0]
      
    # Compute the values for wavenumber in x and y directions
    c = 2.*numpy.pi
    kx = c*numpy.fft.fftfreq(nx, dx)
    ky = c*numpy.fft.fftfreq(ny, dy)    
    
    # Return the final output
    return numpy.meshgrid(kx, ky)