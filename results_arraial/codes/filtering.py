# Title: Filtering
# Author: Nelson Ribeiro Filho / Rodrido Bijani

# Import Python libraries
from __future__ import division
import warnings
import time
import numpy
import auxiliars
import derivative

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
    
    # Conditions for all inputs
    if x.shape != y.shape != data.shape:
        raise ValueError("All inputs must have the same shape!")

    if H == 0.:
        # No continuation will be applied
        res = data
    else:
        # Calculate the wavenumbers
        ky, kx = auxiliars.wavenumber(y, x)
        kcont = numpy.exp((-H) * numpy.sqrt(kx**2 + ky**2))
        result = kcont * numpy.fft.fft2(data)
        res = numpy.real(numpy.fft.ifft2(result))

    # Return the final output
    return res

def reduction(x, y, data, inc, dec, incs=None, decs=None, newinc=None, newdec=None, 
              newincs=None, newdecs=None):
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

    # Conditions for all inputs
    if x.shape != y.shape != data.shape:
        raise ValueError("All inputs must have the same shape!")

    # Induced magnetization
    if incs == None:
        incs = inc
    if decs == None:
        decs = dec
    
    # Reduction to Pole
    if newinc == None:
        newinc = 90.
    if newdec == None:
        newdec = 0.
    if newincs == None:
        newincs = 90.
    if newdecs == None:
        newdecs = 0.
        
    # Step 1 - Calculate the wavenumbers
    # It will return the wavenumbers in x and y directions, in order to calculate the
    # values for magnetization directions in Fourier domains:
    ky, kx = auxiliars.wavenumber(y, x)
    
    # Step 2 - Calcuate the magnetization direction
    # It will return the magnetization directions in Fourier domain for all vector that
    # contain  inclination and declination. All values are complex.
    f0 = auxiliars.theta(inc, dec, kx, ky)
    m0 = auxiliars.theta(incs, decs, kx, ky)
    f1 = auxiliars.theta(newinc, newdec, kx, ky)
    m1 = auxiliars.theta(newincs, newdecs, kx, ky)
       
    # Step 3 - Calculate the filter
    # It will return the result for the reduction filter. However, it is necessary use a
    # condition while the division is been calculated, once there is no zero division.
    with numpy.errstate(divide='ignore', invalid='ignore'):
        operator = (f1 * m1)/(f0 * m0)
    operator[0, 0] = 0.
    
    # Calculate the result by multiplying the filter and the data on Fourier domain
    res = operator*numpy.fft.fft2(data)
    
    # Return the final output
    return numpy.real(numpy.fft.ifft2(res))

def tilt(x, y, data):
    '''
    Return the tilt angle for a potential data on a regular grid.

    Inputs:
    x - numpy 2D array - grid values in x direction
    y - numpy 2D array - grid values in y direction
    data - numpy 2D array - potential data
    
    Output:
    tilt - numpy 2D array - tilt angle for a potential data
    '''
    
    # Stablishing some conditions
    if x.shape != y.shape != data.shape:
        raise ValueError("All inputs must have the same shape!")
    
    # Calculate the horizontal and vertical gradients
    hgrad = derivative.horzgrad(x, y, data)
    derivz = derivative.zderiv(x, y, data, 1)
    
    # Tilt angle calculation
    tilt = auxiliars.my_atan(derivz, hgrad)
    
    # Return the final output
    return tilt

def hyperbolictilt(x, y, data):
    '''
    Return the hyperbolic tilt angle for a potential data.
    
    Inputs:
    x - numpy 2D array - grid values in x direction
    y - numpy 2D array - grid values in y direction
    data - numpy 2D array - potential data
    
    Output:
    hyptilt - numpy 2D array - hyperbolic tilt angle calculated
    '''
    
    # Stablishing some conditions
    if x.shape != y.shape != data.shape:
        raise ValueError("All inputs must have the same shape!")

    # Calculate the horizontal and vertical gradients
    hgrad = derivative.horzgrad(x, y, data)
    diffz = derivative.zderiv(x, y, data, 1)
    
    # Compute the tilt derivative
    hyptilt = auxiliars.my_atan(diffz, hgrad)
    
    # Return the final output
    return numpy.real(hyptilt)

def thetamap(x, y, data):
    '''
    Return the theta map transformed data.
    
    Inputs:
    x - numpy 2D array - grid values in x direction
    y - numpy 2D array - grid values in y direction
    data - numpy 2D array - potential data
    
    Output:
    thetamap - numpy 2D array - thetha map calculated
    '''
    
    # Stablishing some conditions
    if x.shape != y.shape != data.shape:
        raise ValueError("All inputs must have the same shape!")
    
    # Calculate the horizontal and total gradients
    hgrad = derivative.horzgrad(x, y, data)
    tgrad = derivative.totalgrad(x, y, data)
   
    # Return the final output
    return numpy.arccos(hgrad/tgrad)

def pseudograv(x, y, data, inc, dec, incs, decs, rho = 1000., mag = 1.):
    '''
    This function calculates the pseudogravity anomaly transformation due to a total 
    field anomaly grid. It recquires the X and Y coordinates (respectively North and 
    East directions), the magnetic data as a 2D array grid, the values for inclination 
    and declination for the magnetic field and the magnetization of the source.
    
    Inputs:
    x - numpy 2D array - coordinates in X direction
    y - numpy 2D array - coordinates in y direction
    data - numpy 2D array - magnetic data (usually total field anomaly)
    field - numpy 1D array - inclination and declination for the magnetic field
        field[0] -> inclination
        field[1] -> declination
    source - numpy 1D array - inclination and declination for the magnetic source
        source[0] -> inclination
        source[1] -> declination

    Output:
    pgrav - numpy array - pseudo gravity anomaly
    '''
    
    # Conditions (1):
    if x.shape != y.shape != data.shape:
        raise ValueError("All inputs must have the same shape!")
    
    # Conditions (2):
    assert rho != 0., 'Density must not be zero!'
    assert mag != 0., 'Magnetization must not be zero!'
    
    # Conversion for gravity and magnetic data
    G = 6.673e-11
    si2mGal = 100000.0
    t2nt = 1000000000.0
    cm = 0.0000001
        
    # Compute the constant value
    C = G*rho*si2mGal/(cm*mag*t2nt)
    
    # Calculate the wavenumber
    ky, kx = auxiliars.wavenumber(y, x)
    k = (kx**2 + ky**2)**(0.5)
    
    # Computing theta values for the source
    thetaf = auxiliars.theta(inc, dec, kx, ky)
    thetas = auxiliars.theta(incs, decs, kx, ky)
    
    # Calculate the product
    # Here we use the numpy error statement in order to evaluate the zero division
    with numpy.errstate(divide='ignore', invalid='ignore'):
        prod = 1./(thetaf*thetas*k)
    prod[0, 0] = 0.
    
    # Calculate the pseudo gravity anomaly
    res = numpy.fft.fft2(data)*prod
    
    # Converting to mGal as a product by C:
    res *= C
    
    # Return the final output
    return numpy.real(numpy.fft.ifft2(res))

def simple_polynomial(x, y, data):
    '''
    It calculates the regional and residual signal by applying a second-order 
    degree polynomial in order to fit the observed data.
    
    Inputs: 
    xo, yo - numpy array - observation points
    data - numpy array - gravity or magnetic data
    
    Outputs:
    pcoef - list - values of all coefficients
    reg - numpy array - regional signal
    res - numpy array - residual signal
    '''
    
    # Conditions:
    if x.shape != y.shape != data.shape:
        raise ValueError('Observation points must have same shape!')
        
    # Calculate the Jacobian matrix
    mat = numpy.vstack((numpy.ones_like(x), x, y)).T
    # Calculate the polynomial coefficients
    poly = numpy.linalg.solve(numpy.dot(mat.T, mat), 
                              numpy.dot(mat.T, data))
    # Calculate the regional signal
    reg = numpy.dot(mat, poly)
    # Calculate the residual signal
    res = data - reg
    
    # Return the final output
    return reg, res

def robust_polynomial(x, y, data, degree, iterations):
    '''
    It calculates the robust polynomial fitting on regional-residual separation 
    for gravity or magnetic data. It receives the observation points, the data
    and the polynomial degree as well as the number of iterations.
    
    Input:
    x, y - numpy array - observation points
    data - numpy array - gravity or magnetic data
    degree - scalar - degree of polynomial
    iterations - scalar - number of iterations
    
    Output:
    reg - numpy array - regional signal
    res - numpy array - residual signal
    '''
    
    # Computing the time
    timei = time.time()
    
    # Jacobian matrix must be calculated with same rows of observed data and
    # columns equal to (2*N + 1)
    cols = (2*degree) + 1
    
    # Create the Jacobian matrix A
    mat = numpy.zeros((x.size, cols))
    for k in range(cols):
        if k % 2 == 0:
            e = (k/2)
            mat[:,k] = y**e
        else:
            e = (k + 1)/2
            mat[:,k] = x**e
    
    # Solving the linear system in order to calculate the simple fitting
    # data by least squares method
    poly_simple = numpy.linalg.solve(numpy.dot(mat.T, mat), 
                                     numpy.dot(mat.T, data))
    # Calculate the regional and residual by least square
    reg_simple = numpy.dot(mat, poly_simple)
    
    # Initiate the robust polynomial fitting
    # Copy the last result as input
    poly_rob = poly_simple.copy()
    reg_rob = reg_simple.copy()
    
    # Robust polynomial fitting in n iterations
    for i in range(iterations):
        # Calculate the first residual to minimize the difference
        r = data - reg_rob
        s = numpy.median(r)
        #W = numpy.diag(numpy.exp(-((0.6745*r/s)**2.)))
        # Calculate the weight matrix and solve linear system for each iteration
        W = numpy.diag(1./numpy.abs(r + 1.e-5))
        W = numpy.dot(mat.T, W)
        # New robust coefficients 
        poly_rob = numpy.linalg.solve(numpy.dot(W, mat), 
                                    numpy.dot(W, data))
        # Calculate the regional by robust fittin
        reg_rob = numpy.dot(mat, poly_rob)
    
    # Calcualte the residual by robust fitting
    res_rob = data - reg_rob
    
    # Final time
    timef = time.time()
    print 'Time of process (second):', numpy.around((timef - timei), decimals = 3)
    # Return the final output
    return reg_rob, res_rob