# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 16:27:12 2021

@author: arnep
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kalman import state_spacer
import math 

#testing the functions
if __name__ == "__main__":
    #read in the data
    y = pd.read_csv('nile.dat')
    y = np.array(y).astype('float')
    y[50:70] = float("nan")
    period = np.zeros((25,1))   
    period[:] = np.nan
    y = np.append(y, period, axis = 0)
    
    #create state_space object
    Nile_filter = state_spacer(y)
    
    #choose model specification
    simple_model = True
    
    if simple_model:
        #set the function and initialisation of the matrices
        fun = Nile_filter.kalman_llik_diffuse
        matr = Nile_filter.init_matr
        
        #choose which elements to use in MLE
        param_loc = {
            0: {'matrix' : 'Q', 'row' : 0, 'col' : 0} ,
            1: {'matrix' : 'H', 'row' : 0, 'col' : 0} 
                     }
        
        #initialise parameters and filter
        filter_init =  (0), (1e7) 
        param_init= {
            0:  1,
            1:  1
            }
        
        #test time-varying functionality by setting Q in a time-varying way
        Q = np.ones((1,1,len(y)))
        Nile_filter.init_state_matrices( T=None, R=None, Z=None, Q=Q, H=np.array([1]), 
                                        c=None, d=None, states = 1, eta_size = 1)
    
        #estimate MLE parameters
        bnds = ((1, 50000),(1, 50000))
        est =  Nile_filter.ml_estimator_matrix(fun, matr, param_loc, filter_init, param_init, bnds)
    
        #get output
        at, Pt, a, P, v, F, K, _, _ = Nile_filter.kalman_filter(Nile_filter.init_matr,filter_init, y)
        at, Pt, a, P, v, F, K, alpha, V, r, N = Nile_filter.smoother(Nile_filter.init_matr,filter_init, y)

    
        y = y[:70]

        #get output
        at, Pt, a, P, v, F, K, _, _ = Nile_filter.kalman_filter(Nile_filter.init_matr,filter_init, y)
        at, Pt, a, P, v, F, K, alpha, V, r, N = Nile_filter.smoother(Nile_filter.init_matr,filter_init, y)
        
    else:
        #set the function
        fun = Nile_filter.kalman_llik_diffuse
        
        #choose which elements to use in MLE
        param_loc = {
            0: {'matrix' :'Q', 'row' : 0, 'col' : 0} ,
            1: {'matrix' :'Q', 'row' : 1, 'col' : 1} ,
            2: {'matrix' :'H', 'row' : 0, 'col' : 0} ,
            3: {'matrix' :'T', 'row' : 0, 'col' : 0} 
                     }
        
        #initialise parameters and filter
        filter_init =  (0,0), ((1e7,0), (0,1e7))
        param_init= {
            0:  1,
            1:  3,
            2:  5,
            3:  0.5
            }
    
        #set initial matrices
        Nile_filter.init_state_matrices( T=None, R=None, Z=None, Q=None, H= None, 
                                    c=None, d=None, states = 2, eta_size = 2)
        matr = Nile_filter.init_matr
    

        #estimate MLE parameters
        bnds = ((1, 50000),(1, 50000),(1, 50000), (-.999,.999))
        est =  Nile_filter.ml_estimator_matrix(fun, matr, param_loc, filter_init, param_init, bnds)
    

        #get output
        at, Pt, a, P, v, F, K, _, _ = Nile_filter.kalman_filter(Nile_filter.init_matr,filter_init)
        at, Pt, a, P, v, F, K, alpha, V, r, N = Nile_filter.smoother(Nile_filter.init_matr,filter_init)

        y = y[:70]
        at, Pt, a, P, v, F, K, _, _ = Nile_filter.kalman_filter(Nile_filter.init_matr,filter_init)
        at, Pt, a, P, v, F, K, alpha, V, r, N = Nile_filter.smoother(Nile_filter.init_matr,filter_init)
    #plot data and filtered and smoothed values    
    plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(at[:,:].sum(axis=2), label =  'Filtered state')
    plt.plot(alpha[:,:].sum(axis=2), label = 'Smoothed state')
    plt.scatter(range(len(y)), y, alpha = 0.3, label = 'Observations')
    plt.title('Nile volume: observations vs. filtered and smoothed states')
    plt.legend()
    plt.show()
    
    