# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 11:33:24 2021

@author: arnep
"""

import numpy as np
import matplotlib.pyplot as plt
from ekf import eStateSpacer

def lin_fun(x, b):
    """
    linear function of the form y = bx, with x the observation vector and
    b the parameter vector.
    
    Parameters
    ----------
    x : data
    b : parameters

    Returns
    -------
    x*b.transpose()
    
        """
    return  x*b.transpose()

def one_fun(x, b):
    """
    linear function of the form y = b, with x the observation vector and
    b the parameter vector.
    

    Parameters
    ----------
    x : data
    b : parameters

    Returns
    -------
    b

    """
    return b

def square_fun(x, b):
    """
    linear function of the form y = bx², with x the observation vector and
    b the parameter vector.
    
    Parameters
    ----------
    x : data
    b : parameters

    Returns
    -------
    x²*b.transpose()
    """
    return  (x**2)*b.transpose()

def square_der_fun(x, b):
    """
    linear function of the form y = 2bx, with x the observation vector and
    b the parameter vector.
    
    Parameters
    ----------
    x : data
    b : parameters

    Returns
    -------
    2*x*b.transpose()
    """

    return  2*x*b.transpose()


def dummy_fun(x):
    return x


if __name__ == "__main__":
    
    #create data generating process: set parameters and set seed
    n = 400
    Q = np.matrix(4.0)
    H=  np.matrix(0.01)
    T= np.matrix(0.7)
    np.random.seed(10)
    Z = np.matrix(1)
    
    #make noise vectors
    noise_obs  = np.random.normal(0,H,n)
    noise_state  = np.random.normal(0, Q,n)
    
    #create state vector and initialise
    a_array = np.zeros((n,1))
    d_array = 3
    c_array = 0
    a_array[0] = 0
    
    #fill the state vector
    for i in range(1,n):
        a_array[i]= d_array + a_array[i-1]*T.transpose()  + noise_state[i]
    
    #create observation vector
    y = c_array + (a_array **2)*Z.transpose()+ noise_obs.reshape(-1,1) 
    y = np.array(y)
    
    #set a number of observations to nan to demonstrate missing value handling
    y[50:80] = np.nan
    a_array[50:80] = np.nan
    
    period = np.zeros((25,1))   
    period[:] = np.nan
    y = np.append(y, period, axis = 0)

    #create state_space object
    QuadFilter = eStateSpacer(square_fun, square_der_fun, lin_fun, one_fun )
    #QuadFilter = eStateSpacer(lin_fun, one_fun, lin_fun, one_fun )
    #set the function
    kalman_llik = QuadFilter.kalman_llik
    
    #initialise parameters and filter
    filter_init =  (np.sqrt(np.mean(y[~np.isnan(y)]))), (np.std(y[~np.isnan(y)]))
    param_init=  (0,1, 1,0,0)
    param_init=  (1, 1, 1)
    param_init=  (0, 1, 1,0)

    #set initial matrices
    QuadFilter.init_matrices( T= T, R=None, Z=None, Q=Q, H= H, 
                                c=None, d=None, states = 1, eta_size = 1)
    
    QuadFilter.matr['T'][0,0] = np.nan
    QuadFilter.matr['Q'][0,0] = np.nan
    QuadFilter.matr['H'][0,0] = np.nan
    QuadFilter.matr['d'][0,0] = np.nan
    
    #estimate MLE parameters
    bnds = ((-.999,.999),(.00001, 1000),(0.00001, 1000),(-5,5))

    QuadFilter.fit(y, optim_fun=kalman_llik,
                            filter_init=filter_init, param_init=param_init, bnds=bnds)
     
    #get output
    o = QuadFilter.smoother(y, filter_init)
    output, errors = o['output'], o['errors']
    
    
    #test the simulation smoother
    nsim =10
    simulations = QuadFilter.simulation_smoother(y, filter_init, nsim )
    
    for i in range(nsim):
        plt.plot(simulations[:,:,i],c='grey', alpha=0.2)
    for i in range(output['alpha'].shape[2]):
        plt.plot(output['alpha'][:,:,i], c='orange', label= 'Smoothed state')
    plt.title('Simulation smoother')
    plt.show()


    
    #do not plot first observations 
    first = 1
    #plot data and filtered and smoothed values    
    plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(output["at"][first:,:].sum(axis=2)**2, label =  'Filtered signal')
    plt.plot(output["alpha"][first:,:].sum(axis=2)**2, label = 'Smoothed signal')
    plt.scatter(range(len(y) - first),y[first:] ,  alpha = 0.3, label = 'Observation')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(output["at"][first:,:].sum(axis=2), label =  'Filtered state')
    plt.plot(output["alpha"][first:,:].sum(axis=2), label = 'Smoothed state')
    plt.scatter(range(len(a_array)-first), np.abs(a_array[first:]) ,  alpha = 0.3, label = 'State')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(errors["epsilon_hat"][first:,:].sum(axis=2), label =  'Observation errors')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(errors["eta_hat"][first:,:].sum(axis=2), label =  'State errors')
    plt.legend()
    plt.show()
    
