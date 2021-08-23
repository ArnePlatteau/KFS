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
    np.random.seed(10)
    y[50:70] = float("nan")
    period = np.zeros((25,1))   
    period[:] = np.nan
    period_pre = np.zeros((30,1))   
    period_pre[:] = np.nan

    y = np.append(y, period, axis = 0)
    y = np.append(period_pre, y, axis = 0)

    conf =0.9
    
    #number of simulations in simulation smoother
    nsim = 300
    
    #create state_space object
    Nile_filter = state_spacer()
    
    #choose model specification
    simple_model = True
    
    if simple_model:
        #set the function and initialisation of the matrices
        kalman_llik = Nile_filter.kalman_llik_diffuse
        
        #initialise parameters and filter
        filter_init =  (0), (1e7) 
        
        
        param_init = np.array((100,100))
        
        #test time-varying functionality by setting Q in a time-varying way
        Q = np.ones((1,1,len(y)))
        Nile_filter.init_matrices( T=None, R=None, Z=None, Q=Q, H=None, 
                                        c=None, d=None, y=y, states = 1, eta_size = 1)

        Nile_filter.matr['Q'][0,0] = np.nan
        Nile_filter.matr['H'][0,0] = np.nan

        #estimate MLE parameters
        bnds = ((0.1, 50000),(0.1, 50000))

        
    else:
        #set the function
        kalman_llik = Nile_filter.kalman_llik_diffuse
        T_stationary = np.matrix(0.0507867)
        Q_stationary = 4000.001901735713
   #     var_stationary = np.linalg.inv(np.eye(1) - np.kron(T_stationary,T_stationary))*(1*4000.001901735713*1).reshape((-1, 1), order="F")
        var_stationary = (np.linalg.inv(np.eye(1) - np.kron(T_stationary,T_stationary))*(1*4000.001901735713*1))[0,0]
        #initialise parameters and filter
        filter_init =  (0,0), ((var_stationary,0), (0,1e7))

        param_init= (0.5, 1, 3, 5)

        #set initial matrices
        Nile_filter.init_matrices( T=None, R=None, Z=None, Q=None, H= None, 
                                    c=None, d=None, y= y, states = 2, eta_size = 2)
        
        Nile_filter.matr['T'][0,0] = np.nan
        Nile_filter.matr['Q'][1,1] = np.nan
        Nile_filter.matr['H'][0,0] = np.nan
        Nile_filter.matr['Q'][0,0] = np.nan
        
        bnds = ((-.999,.999), (4000, 5000), (12, 16),(180, 200))

    #estimate MLE parameters
    Nile_filter.fit(y, kalman_llik=kalman_llik,
                    filter_init=filter_init, param_init=param_init, bnds=bnds)


    o = Nile_filter.smoother(y, filter_init)
    output, errors = o['output'], o['errors']

    #get output
    o = Nile_filter.kalman_smoother_CI(y, filter_init,conf=conf)
    output, errors = o['output'], o['errors']
        
   
    #test the simulation smoother    
    simulations = Nile_filter.simulation_smoother_2(y, filter_init, nsim)
    
    #plot simulated paths and actual alpha
    for i in range(nsim):
        plt.plot(simulations[:,:,i],c='grey', alpha=0.2)
    for i in range(output['alpha'].shape[2]):
        plt.plot(output['alpha'][:,:,i], c='orange', label= 'Smoothed state')
    plt.title('Simulation smoother')
    plt.show()
    
    #plot data and filtered and smoothed values    
    plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(output["at"][:,:].sum(axis=2), label =  'Filtered state')
    plt.plot(output['CI_filtered_' + str(conf)  + "_upper"][:,:].sum(axis=2), label =  'CI_filtered_' + str(conf)  + "_upper", c='blue')
    plt.plot(output['CI_filtered_' + str(conf)  + "_lower"][:,:].sum(axis=2), label =  'CI_filtered_' + str(conf)  + "_lower", c='blue')
    plt.plot(output["alpha"][:,:].sum(axis=2), label = 'Smoothed state')
    plt.plot(output['CI_smoothed_' + str(conf)  + "_upper"][:,:].sum(axis=2), label =  'CI_smoothed_' + str(conf)  + "_upper", c='orange')
    plt.plot(output['CI_smoothed_' + str(conf)  + "_lower"][:,:].sum(axis=2), label =  'CI_smoothed_' + str(conf)  + "_lower", c='orange')
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
