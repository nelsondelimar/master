from __future__ import division
import warnings
import numpy

def my_poly(x, y, data):
    '''
    It calculates the regional and residual signal by applying a second-order degree polynomial in order to fit the observed data.
    
    Inputs: 
    xo, yo - numpy array - observation points
    data - numpy array - gravity or magnetic data
    
    Outputs:
    poly - list - values of all coefficients
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
    return poly, reg, res

def my_robust_poly(x, y, data, degree = 2, iterations = 20):
    '''
    It calculates the robust polynomial fitting on regional-residual separation for gravity or magnetic data. 
    It receives the observation points, the data and the polynomial degree as well as the number of iterations.
    
    Input:
    x, y - numpy array - observation points
    data - numpy array - gravity or magnetic data
    degree - scalar - degree of polynomial
    iterations - scalar - number of iterations
    
    Output:
    reg - numpy array - regional signal
    res - numpy array - residual signal
    '''
    # Jacobian matrix must be calculated with same rows of observed data and columns equal to (2*N + 1)
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
        # Calculate the weight matrix and solve linear system for each iteration
        W = numpy.diag(1./numpy.abs(r + 1.e-10))
        W = numpy.dot(mat.T, W)
        # New robust coefficients 
        poly_rob = numpy.linalg.solve(numpy.dot(W, mat), 
                                    numpy.dot(W, data))
        # Calculate the regional by robust fitting
        reg_rob = numpy.dot(mat, poly_rob)    
    # Calcualte the residual by robust fitting
    res_rob = data - reg_rob    
    # Return the final output
    return poly_rob, reg_rob, res_rob