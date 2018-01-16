# -----------------------------------------------------------------------------------
# Title: Data Filtering
# Author: Nelson Ribeiro Filho
# Description: Source codes for potential field filtering
# Collaborator: Rodrigo Bijani
# -----------------------------------------------------------------------------------

# Import Python libraries
from __future__ import division
import warnings
import numpy as np
import warnings
import auxiliars as aux

def statistical(data, unit=None):
    
    '''
    A statistical function that calculates the minimum and maximum values for a simple 
    dataset and also its mean values and variations. The dataset can be a 1D or a 2D
    numpy array.
    
    Input:
    data - numpy array - data set as a vector
    unit - string - data unit
    
    Outputs:
    datamin - float - minimun value for the data
    datamax - float - maximun value for the data
    datamed - float - mean value for all dataset
    datavar - float - variation for all dataset
    
    '''
    
    assert data.size > 1, 'Data set must have more than one element!'
    
    datamin = np.min(data)
    datamax = np.max(data)
    datamed = np.mean(data)
    datavar = datamax - datamin
    
    if (unit != None):
    	print 'Minimum:    %5.4f' % datamin, unit
        print 'Maximum:    %5.4f' % datamax, unit
        print 'Mean value: %5.4f' % datamed, unit
        print 'Variation:  %5.4f' % datavar, unit
    else:
        print 'Minimum:    %5.4f' % datamin
        print 'Maximum:    %5.4f' % datamax
        print 'Mean value: %5.4f' % datamed
        print 'Variation:  %5.4f' % datavar
        
    return datamin, datamax, datamed, datavar

def continuation(x, y, data, H):
    
    '''
    This function compute the upward or downward continuation for a potential field 
    data, which can be gravity or magnetic signal. The value for H represents the 
    level which the data will be continuated. If H is positive, the continuation is 
    upward, because Dz is greater than 0 and the exponential is negative; otherwise, 
    if H is negative, the continuation is downward.
    
    Input:
    x - numpy 2D array - observation points on the grid in X direction
    y - numpy 2D array - observation points on the grid in Y direction
    data - 2D array - gravity or magnetic data
    H - float - value for the new observation level
    
    '''
    assert H != 0., 'Height must be different of zero!'
    
    # calculate the wavenumbers
    kx, ky = wavenumber(x, y)
    
    if H > 0.:
        #print ('H is positive. Continuation is Upward!')
        kcont = np.exp((-H) * np.sqrt(kx**2 + ky**2))
        result = kcont * np.fft.fft2(data)
    elif H < 0.:
        #print ('H is negative. Continuation is Downward!')
        kcont = np.exp((-H) * np.sqrt(kx**2 + ky**2))
        result = kcont * np.fft.fft2(data)

    return np.real(np.fft.ifft2(result))

def reduction(x, y, data, oldf, olds, newf, news):
    
    '''
    Return the reduced potential data giving the new directions for the geomagnetic
    field and source magnetization. Its based on Blakely (1996).
    
    Inputs: 
    x - numpy 2D array - coordinate at X
    y - numpy 2D array - coordinate at Y
    data - numpy 2D array - magnetic data set (usually total field anomaly)
    oldf - numpy 1D array - vector with old field directions
    olds - numpy 1D array - vector with old source directions
    newf - numpy 1D array - vector with new field directions
    news - numpy 1D array - vector with new source directions
    
    - The last four vector are discplaced as : v = [inc, dec]
    
    Output:
    res - numpy 2D array - result by using reduction filter
    
    Ps. This filter is very useful for values of incination greater than +/- 15 deg.
    '''

    # Conditions for X and Y grids
    assert x.shape == y.shape, 'Grid in X and grid in Y must have same dimension!'
    assert x.shape == data.shape, 'Grid in X and Data must have same dimension!'
    assert y.shape == data.shape, 'Grid in X and Data must have same dimension!'
    
    # Conditions for all direction vectors
    assert olfd.size == newf.size, 'Vector must have only inclination and declination!'
    assert olfs.size == news.size, 'Vector must have only inclination and declination!'
    
    # Step 1 - Calculate the wavenumbers
    # It will return the wavenumbers in x and y directions, in order to calculate the
    # values for magnetization directions in Fourier domains:
    kx, ky = wavenumber(x, y)
    
    # Step 2 - Calcuate the magnetization direction
    # It will return the magnetization directions in Fourier domain for all vector that
    # contain  inclination and declination. All values are complex.
    f0 = theta(oldf, kx, ky)
    m0 = theta(olds, kx, ky)
    f1 = theta(newf, kx, ky)
    m1 = theta(news, kx, ky)
       
    # Step 3 - Calculate the filter
    # It will return the result for the reduction filter. However, it is necessary use a
    # condition while the division is been calculated, once there is no zero division.
    with np.errstate(divide='ignore', invalid='ignore'):
        operator = (f1 * m1)/(f0 * m0)
    operator[0, 0] = 0.
    
    # Calculate the result by multiplying the filter and the data on Fourier domain
    res = operator*np.fft.fft2(data)
    
    # Return the final output
    return np.real(np.fft.ifft2(res))

def xderiv(x, y, data, n):
    
    '''
    Return the horizontal derivative in x direction for n order in Fourier domain.
    
    Inputs:
    x - numpy 2D array - grid values in x direction
    y - numpy 2D array - grid values in y direction
    data - numpy 2D array - potential data
    n - float - order of the derivative
    
    Output:
    xder - numpy 2D array - derivative in x direction
    '''
    
    # Stablishing some conditions
    assert x.shape == data.shape, 'Grid in X and data must have the same shape!'
    assert y.shape == data.shape, 'Grid in Y and data must have the same shape!'
    
    # Condition for the order of the derivative
    assert n > 0., 'Order of the derivative must be positive and nonzero!'
    
    # Calculate the wavenuber in x direction
    kx, _ = wavenumber(x,y)
    
    # Apply the Fourier transform
    xder = np.fft.fft2(data)*((kx*1j)**(n))
    
    # Return the final output
    return np.real(np.fft.ifft2(xder))

def yderiv(x, y, data, n):
    
    '''
    Return the horizontal derivative in y direction for n order in Fourier domain.
    '''
    
    # Stablishing some conditions
    assert x.shape == data.shape, 'Grid in X and data must have the same shape!'
    assert y.shape == data.shape, 'Grid in Y and data must have the same shape!'
    
    # Condition for the order of the derivative
    assert n > 0., 'Order of the derivative must be positive and nonzero!'
    
    # Calculate the wavenuber in y direction
    _, ky = wavenumber(x,y)
    
    # Apply the Fourier transform
    yder = np.fft.fft2(data)*((ky*1j)**(n))
    
    # Return the final output
    return np.real(np.fft.ifft2(yder))

def zderiv(x, y, data, n):
    
    '''
    Return the vertical derivative in z direction for n order in Fourier domain.
    '''
    
    # Stablishing some conditions
    assert x.shape == data.shape, 'Grid in X and data must have the same shape!'
    assert y.shape == data.shape, 'Grid in Y and data must have the same shape!'
    
    # Condition for the order of the derivative
    assert n > 0., 'Order of the derivative must be positive and nonzero!'    

    # Calculate the wavenuber in z direction
    kx, ky = wavenumber(x,y)
    
    # Apply the Fourier transform
    zder = np.fft.fft2(data)*(np.sqrt(kx**2 + ky**2)**(n))
    
    # Return the final output
    return np.real(np.fft.ifft2(zder))

def horzgrad(x, y, data):
    
    '''
    Return the horizontal gradient amplitude (HGA) for a potential data on a regular 
    grid. All calculation is done by using Fourier domain.
    '''
    
    # Stablishing some conditions
    assert x.shape == data.shape, 'Grid in X and data must have the same shape!'
    assert y.shape == data.shape, 'Grid in Y and data must have the same shape!'
    
    # Computes the horizontal derivatives
    derivx = xderiv(x, y, data, 1)
    derivy = yderiv(x, y, data, 1)
    
    # Calculates the total gradient
    hga = (derivx**2 + derivy**2)**(0.5)
    
    # Return the final output
    return hga

def totalgrad(x, y, data):
    
    '''
    Return the total gradient amplitude (TGA) for a potential data on a regular grid.
    '''
    
    # Stablishing some conditions
    assert x.shape == data.shape, 'Grid in X and data must have the same shape!'
    assert y.shape == data.shape, 'Grid in Y and data must have the same shape!'

    # Calculates the x derivative
    derivx = xderiv(x, y, data, 1)
    # Calculates the y derivative
    derivy = yderiv(x, y, data, 1)
    # Calculates the z derivative
    derivz = zderiv(x, y, data, 1)

    # Calculates the total gradient
    tga = (derivx**2 + derivy**2 + derivz**2)**(0.5)
    
    # Return the final output
    return tga

def tilt(x, y, data):
    
    '''
    Return the tilt angle for a potential data on a regular grid.
    '''
    
    # Stablishing some conditions
    assert x.shape == data.shape, 'Grid in X and data must have the same shape!'
    assert y.shape == data.shape, 'Grid in Y and data must have the same shape!'
    
    # Calculate the horizontal and vertical gradients
    hgrad = horzgrad(x, y, data)
    derivz = zderiv(x, y, data, 1)
    
    # Tilt angle calculation
    tilt = np.arctan2(derivz, hgrad)
    
    # Return the final output
    return tilt

def thetamap(x, y, data):

    '''
    Return the theta map transformed data.
    '''
    
    # Stablishing some conditions
    assert x.shape == data.shape, 'Grid in X and data must have the same shape!'
    assert y.shape == data.shape, 'Grid in Y and data must have the same shape!'
    
    # Calculate the horizontal and total gradients
    hgrad = horzgrad(x, y, data)
    tgrad = totalgrad(x, y, data)
   
    # Return the final output
    return (hgrad/tgrad)

def hyperbolictilt(x, y, data):
    
    '''
    Return the hyperbolic tilt angle for a potential data.
    '''
    
    # Stablishing some conditions
    assert x.shape == data.shape, 'Grid in X and data must have the same shape!'
    assert y.shape == data.shape, 'Grid in Y and data must have the same shape!'

    # Calculate the horizontal and vertical gradients
    hgrad = horzgrad(x, y, data)
    derivz = zderiv(x, y, data, 1)
    
    # Compute the tilt derivative
    hyptilt = np.arctan2(derivz, hgrad)
    
    # Return the final output
    return np.real(hyptilt)

def theta(angle, u, v):
    
    '''
    Return the operators for magnetization and field directions.
    
    Inputs:
    angle - numpy 1D array - inclination and declination
    u - float - number of points in x direction
    v - float - number of points in y direction
    '''
    
    # Calculate the modulus for k value. In this case: k = kz
    k = (u**2 + v**2)**(0.5)
    
    # Defines inclination and declination:
    inc, dec = angle[0], angle[1]
    
    # Calcutaing the projections
    x, y, z = aux.dircos(inc, dec) 
    theta = z + ((x*u + y*v)/k)*1j
    
    # Return the final output:
    return theta

def wavenumber(x, y):
    '''
    Return the wavenumbers in X and Y directions
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
    c = 2.*np.pi
    kx = c*np.fft.fftfreq(nx, dx)
    ky = c*np.fft.fftfreq(ny, dy)    
    
    # Return the final output
    return np.meshgrid(kx, ky)

def pseudograv(x, y, data, field, source, rho, mag):
    '''
    This function calculate the pseudogravity anomaly due to a total field anomaly mag. It recquires the X and Y coordinates (North and East directions), the magnetic data, the values for inclination and declination for the magnetic field and the magnetization of the source.
    
    Inputs:
    x, y - numpy array - 
    data - numpy array - magnetic field anomaly
    field -numpy array - inclination and declination for the magnetic field
        field[0] -> inclination
        field[1] -> declination
    source -numpy array - inclination and declination for the magnetic source
        source[0] -> inclination
        source[1] -> declination
    
    
    Output:
    pgrav - numpy array - pseudo gravity anomaly
    '''
    
    # Conditions:
    assert rho != 0., 'Density must not be zero!'
    assert mag != 0., 'Density must not be zero!'
    
    # Conversion for gravity and magnetic data
    G = 6.673e-11
    si2mGal = 100000.0
    t2nt = 1000000000.0
    cm = 0.0000001
        
    # Compute the constant value
    C = G*rho*si2mGal/(cm*mag*t2nt)
    
    # Calculate the wavenumber
    kx, ky = wavenumber(x, y)
    k = (kx**2 + ky**2)**(0.5)
    
    # Computing theta values for the source
    thetaf = theta(field, kx, ky)
    thetas = theta(source, kx, ky)
    
    # Calculate the product
    with np.errstate(divide='ignore', invalid='ignore'):
        prod = 1./(thetaf*thetas*k)
    prod[0, 0] = 0.
    
    res = np.fft.fft2(data)*prod
    
    return C*np.real(np.fft.ifft2(res))

def cccoef(data1, data2):
    
    '''
    Returns the simple crosscorrelation coefficient between two data sets, 
    which can or a single 1D array or a N-dimensional data set. It is very
    important that both data sets have the same dimension, otherwise it will
    runnig the code error.
    
    Inputs:
    data1 - numpy array - first dataset
    data2 - numpy array - second dataset
    
    Output:
    res - scalar - cross correlation coefficient
    '''
    
    # Stablish some conditions
    assert data1.shape[0] == data2.shape[1], 'Both dataset must have the same dimension!'
    assert data1.size == data2.size, 'Both dataset must have the same number of elements!'
    
    # Calculate the simple mean
    # For the first dataset
    mean1 = data1.mean()
    # For the second dataset
    mean2 = data2.mean()
       
    # Formulation
    numerator = np.sum((data1 - mean1)*(data2 - mean2))
    den1 = np.sum((data1 - mean1)**2)
    den2 = np.sum((data2 - mean2)**2)
    res = numerator/np.sqrt(den1*den2)
    
    return res