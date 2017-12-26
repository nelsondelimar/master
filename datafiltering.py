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

def continuation(x, y, data, H):
    
    '''
    This function compute the upward or downward continuation for a potential field data,
    which can be gravity or magnetic signal. The value for H represents the level which 
    the data will be continuated. If H is positive, the continuation is upward, because Dz 
    is greater than 0 and the exponential is negative; otherwise, if H is negative, the 
    continuation is downward.
    
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
        print ('H is positive. Continuation is Upward!')
        kcont = np.exp((-H) * np.sqrt(kx**2 + ky**2))
        result = kcont * np.fft.fft2(data)
    elif H < 0.:
        print ('H is negative. Continuation is Downward!')
        kcont = np.exp((-H) * np.sqrt(kx**2 + ky**2))
        result = kcont * np.fft.fft2(data)

    return np.real(np.fft.ifft2(result))

def reduction(x, y, data, oldf, olds, newf, news):
    '''
    Return the reduced potential data giving the new directions for the geomagnetic field and source magnetization.
    '''
    # 1 - Verificar o tamanho do grid e se precisa expandir
    # 2 - Calcular os valores de nx, ny, dx e dy
    kx, ky = wavenumber(x, y)
    f0 = theta(oldf, kx, ky)
    m0 = theta(olds, kx, ky)
    f1 = theta(newf, kx, ky)
    m1 = theta(news, kx, ky)
       
    with np.errstate(divide='ignore', invalid='ignore'):
        operator = (f1 * m1)/(f0 * m0)
    operator[0, 0] = 0.
    res = operator*np.fft.fft2(data)
    return np.real(np.fft.ifft2(res))

def xderiv(x, y, data, n):
    '''
    Return the horizontal derivative in x direction for n order.
    '''
    
    assert n > 0., 'Order of the derivative must be positive and nonzero!'
    
    kx, _ = wavenumber(x,y)
    
    xder = np.fft.fft2(data)*((kx*1j)**(n))
    
    return np.real(np.fft.ifft2(xder))

def yderiv(x, y, data, n):
    '''
    Return the horizontal derivative in y direction for n order.
    '''
    
    assert n > 0., 'Order of the derivative must be positive and nonzero!'
    
    _, ky = wavenumber(x,y)
    
    yder = np.fft.fft2(data)*((ky*1j)**(n))
    
    return np.real(np.fft.ifft2(yder))

def zderiv(x, y, data, n):
    '''
    Return the vertical derivative in z direction for n order.
    '''
    
    assert n > 0., 'Order of the derivative must be positive and nonzero!'
    
    kx, ky = wavenumber(x,y)
    
    zder = np.fft.fft2(data)*(np.sqrt(kx**2 + ky**2)**(n))
    
    return np.real(np.fft.ifft2(zder))

def totalgrad(x, y, data, xderivative, yderivative, zderivative):
    '''
    Return the total gradient amplitude (TGA) for a potential data on a regular grid.
    '''
    
    if xderivative, yderivative and zderivative is None:
        dfdx = xderiv(x, y, data, 1)
        dfdy = yderiv(x, y, data, 1)
        dfdz = zderiv(x, y, data, 1)
        tga = (dfdx**2 + dfdy**2 + dfdz**2)**(0.5)    
    else:
        tga = (xderivative**2 + yderivative**2 + zderivative**2)**(0.5)
    # Return the final output
    return tga

def tilt(x, y, data):
    '''
    Return the tilt angle for a potential data.
    '''
    xdiff = xderiv(x, y, data, 1)
    ydiff = yderiv(x, y, data, 1)
    zdiff = zderiv(x, y, data, 1)
    # Return the final output
    return np.arctan2(np.sqrt(xdiff**2 + ydiff**2), zdiff)

def theta(angle, u, v):
    '''
    Return the operators for magnetization and field directions
    '''
    
    k = (u**2 + v**2)**(0.5)
    
    inc, dec = angle[0], angle[1]
    # Calcutaing the projections
    x, y, z = aux.dircos(inc, dec) 
    theta = z + ((x*u + y*v)/k)*1j
    return theta

def wavenumber(x, y):
    '''
    Return the wavenumbers in X and Y directions
    '''
    
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
      
    c = 2.*np.pi
    kx = c*np.fft.fftfreq(nx, dx)
    ky = c*np.fft.fftfreq(ny, dy)    
    return np.meshgrid(kx, ky)
    