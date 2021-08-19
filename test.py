# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 16:27:12 2021

@author: arnep
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kalman import state_spacer
import time

#testing the functions
if __name__ == "__main__":
    start_time = time.time()
    #read in the data
    y = pd.read_csv('nile.dat')
    y = np.array(y).astype('float')
    y[50:70] = float("nan")
    period = np.zeros((25,1))   
    period[:] = np.nan
    y = np.append(y, period, axis = 0)
    
    #number of simulations in simulation smoother
    nsim = 200
    
    #create state_space object
    Nile_filter = state_spacer()
    
    #choose model specification
    simple_model = False
    
    if simple_model:
        #set the function and initialisation of the matrices
        kalman_llik = Nile_filter.kalman_llik_diffuse
        
        #initialise parameters and filter
        filter_init =  (0), (1e7) 
        
        
        param_init = np.array((100,100))
        
        #test time-varying functionality by setting Q in a time-varying way
       # T = np.ones((1,1,len(y)))
        Nile_filter.init_matrices( T=None, R=None, Z=None, Q=None, H=None, 
                                        c=None, d=None, y=y, states = 1, eta_size = 1)

        Nile_filter.matr['Q'][0,0] = np.nan
        Nile_filter.matr['H'][0,0] = np.nan


        #estimate MLE parameters
        bnds = ((0.1, 50000),(0.1, 50000))
        Nile_filter.fit(y, kalman_llik=kalman_llik,
                        filter_init=filter_init, param_init=param_init, bnds=bnds)

        
    else:
        #set the function
        kalman_llik = Nile_filter.kalman_llik_diffuse
                
        #initialise parameters and filter
        filter_init =  (0,0), ((1e7,0), (0,1e7))

        param_init= (0.5, 1, 3, 5)

        #set initial matrices
        Nile_filter.init_matrices( T=None, R=None, Z=None, Q=None, H= None, 
                                    c=None, d=None, y= y, states = 2, eta_size = 2)
        
        Nile_filter.matr['T'][0,0] = np.nan
        Nile_filter.matr['Q'][1,0] = np.nan
        Nile_filter.matr['H'][0,0] = np.nan
        Nile_filter.matr['Q'][0,0] = np.nan
        
        bnds = ((-.999,.999), (4000, 5000), (12, 16),(180, 200))

    #estimate MLE parameters
    Nile_filter.fit(y, kalman_llik=kalman_llik,
                    filter_init=filter_init, param_init=param_init, bnds=bnds)

    #get output
    o = Nile_filter.smoother(y, filter_init)
    output, errors = o['output'], o['errors']
        


    #test the simulation smoother
    simulations = Nile_filter.simulation_smoother(y, filter_init, nsim)
    
    #plot simulated paths and actual alpha
    for i in range(nsim):
        plt.plot(simulations[:,:,i],c='grey')
    for i in range(output['alpha'].shape[2]):
        plt.plot(output['alpha'][:,:,i])
    plt.show()




    #plot data and filtered and smoothed values    
    plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(output["at"][:,:].sum(axis=2), label =  'Filtered state')
    plt.plot(output["alpha"][:,:].sum(axis=2), label = 'Smoothed state')
    plt.scatter(range(len(y)), y, alpha = 0.3, label = 'Observations')
    plt.title('Nile volume: observations vs. filtered and smoothed states')
    plt.legend()
    plt.show()
    
    
    #plot data and filtered and smoothed values    
    plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(errors["epsilon_hat"][:,:].sum(axis=2))
    plt.title('Epsilon hat')
    plt.show()
    
    #plot data and filtered and smoothed values    
    plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(errors["eta_hat"][:,:].sum(axis=2))
    plt.title('Eta hat')
    plt.show()

    Nile_filter.save_json("Nile_filter.json")
    dummy = state_spacer()
    dummy.load_json("Nile_filter.json")
    
    stop_time = time.time()
    print('script time: ' + str(stop_time - start_time))
    