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
#import pandas as pd
#import matplotlib.pyplot as plt
# Import my libraries
import warnings
import gravmag as gm
import auxiliars as aux



def continuation(data, H):
    
    '''
    This function compute the upward or downward continuation for a potential field data. Data can be a gravity
    signal - disturbance or Bouguer anomaly, for example - or a magnetic signal - total field anomaly.
    The value for H represents the level which the data will be continuated. If H is positive, the continuation 
    is upward, because Dz is greater than 0 and the exponential is negative; otherwise, if H is negative, the
    continuation is downward.
    
    Input:
    data - numpy 1D or 2D array - gravity or magnetic data
    H - float - value for the new observation level
    
    '''
    
    if H == 0.:
        print 'No continuation is applied!'
    elif H > 0.:
        print 'H is positive. Continuation is Downward!'
    else:
        print 'H is negative. Continuation is Upward!'
        
    return x

def reduction(x, y, data, field0, source0, field1, source1):
    '''
    '''
    # 1 - Verificar o tamanho do grid e se precisa expandir
    # 2 - Calcular os valores de nx, ny, dx e dy
    #nx = data.shape[0]
    #ny = data.shape[1]
    
    # Entra X e Y mas nao sao usados, verificar depois como fazer 
    # para entrar somente com eespcaamento
    #kx = 2.0*np.pi* np.fft.fftfreq(nx, dx)
    #ky = 2.0*np.pi* np.fft.fftfreq(ny, dy)
    #KX, KY = np.meshgrid(kx, ky)
    kx, ky = wavenumber(x, y)
    #print(kx[0])
    #print(ky)
    #print(KX[0,0])
    #print(KY[0,0])
    
    # calcular thetaM e thetaF - AQUI TAMBEM VAI ENTRAR OS VALORES DE KX E KY DO GRID
    theta_f0 = theta(field0, kx, ky)
    theta_m0 = theta(source0, kx, ky)
    theta_f1 = theta(field1, kx, ky)
    theta_m1 = theta(source1, kx, ky)
    
    #print(theta_m1)
    #print(theta_f1)
    
    # CALCULANDO A TRANSFORMACAO
    # Numerator and denominator including on the filter
    num = theta_f0 * theta_m0
    den = theta_f1 * theta_m1
    
    with np.errstate(divide='ignore', invalid='ignore'):
        operator = (den/num)
    
    operator[0, 0] = 0.
        
    res = operator*np.fft.fft2(data)
    
    return np.real(np.fft.ifft2(res))

def theta(angle, u, v):
    '''
    '''
    # calculando o modulo
    k = np.sqrt(u**2 + v**2)
    
    inc, dec = angle[0], angle[1]
    x, y, z = aux.dircos(inc, dec) 
        
    # calculando as projecoes
    #d = (x*kx + y*ky)/k
    theta = z + ((x*u + y*v)/k)*1j
    
    return theta

def wavenumber(x, y):
    #nx = data.shape[0]
    #ny = data.shape[1]
    
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == y.shape[1]
    
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
    