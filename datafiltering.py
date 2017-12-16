# -----------------------------------------------------------------------------------
# Title: Data Filtering
# Author: Nelson Ribeiro Filho
# Description: Source codes for potential field filtering
# Collaborator: Rodrigo Bijani
# -----------------------------------------------------------------------------------

# Import Python libraries
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
# Import my libraries
import auxiliars as aux
import gravmag as gm

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
        x = 1.
        print 'None'
    elif H > 0.:
        x = 2.
        print 'Downward'
    elif:
        x = 3.
        print 'Upward'
        
    return x