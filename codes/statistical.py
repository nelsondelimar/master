from __future__ import division
import warnings
import numpy
import warnings
from codes import auxiliars

def my_analysis(data, unit = '(No Unit)'):  
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
    
    print ('Minimum:    %5.4f' % (datamin), unit)
    print ('Maximum:    %5.4f' % (datamax), unit)
    print ('Mean value: %5.4f' % (datamed), unit)
    print ('Variation:  %5.4f' % (datavar), unit)
    # Return the final analysis    
    return datamin, datamax, datamed, datavar

def my_correlation_coef(data1, data2):
    
    '''
    It returns the simple crosscorrelation coefficient between two data sets, which 
    can or a single 1D array or a N-dimensional data set. It is very important that 
    both data sets have the same dimension, otherwise it will runnig the code error.
    
    Inumpyuts:
    data1 - numpy array - first dataset
    data2 - numpy array - second dataset
    
    Output:
    res - scalar - cross correlation coefficient
    '''
    
    # Stablishing some conditions
    if data1.shape != data2.shape :
        raise ValueError("All inumpyuts must have same shape!")        
    # Formulation
    numerator = numpy.sum((data1 - data1.mean())*(data2 - data2.mean()))
    den1 = numpy.sum((data1 - data1.mean())**2)
    den2 = numpy.sum((data2 - data2.mean())**2)   
    #It calculates the cross correlation coefficient
    res = numerator/numpy.sqrt(den1*den2)    
    # Return the final output
    return res