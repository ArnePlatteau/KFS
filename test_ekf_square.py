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

#create data generating process: set parameters and set seed
n = 200
Q = np.matrix(4.0)
H=  np.matrix(0.01)
T= np.matrix(0.7)
np.random.seed(0)
Z = np.matrix(1)

#make noise vectors
noise_obs  = np.random.normal(0,H,n)
noise_state  = np.random.normal(0, Q,n)

#create state vector and initialise
a_array = np.zeros((n,1))
a_array[0] = 0

#fill the state vector
for i in range(1,n):
    a_array[i]= a_array[i-1]*T.transpose()  + noise_state[i]

#create observation vector
y = (a_array**2)*Z.transpose() + noise_obs.reshape(-1,1)
y = np.array(y)

#set a number of observations to nan to demonstrate missing value handling
y[50:80] = np.nan
a_array[50:80] = np.nan


#create state_space object
QuadFilter = eStateSpacer(square_fun, square_der_fun, lin_fun, one_fun )

#set the function
fun = QuadFilter.kalman_llik_diffuse

#choose which elements to use in MLE
param_loc = {
    0: {'matrix' :'Q', 'row' : 0, 'col' : 0} ,
    1: {'matrix' :'H', 'row' : 0, 'col' : 0} ,
             }

#initialise parameters and filter
filter_init =  (10), (1e7)
param_init= {
    0:  1,
    1:  1,
    }

#set initial matrices
QuadFilter.init_state_matrices( T= T, R=None, Z=None, Q=Q, H= H, 
                            c=None, d=None, states = 1, eta_size = 1)


#estimate MLE parameters
bnds = ((1, 50000),(1, 50000) )
QuadFilter.fit(y, fun, param_loc, filter_init, param_init, bnds)

#get output
output = QuadFilter.kalman_filter(y, filter_init)
output_smooth = QuadFilter.smoother(y, filter_init)


#do not plot first observations 
first = 1
#plot data and filtered and smoothed values    
plt.figure(figsize=(10, 6), dpi=200)
plt.plot(output_smooth["at"][first:,:].sum(axis=2)**2, label =  'Filtered signal')
plt.plot(output_smooth["alpha"][first:,:].sum(axis=2)**2, label = 'Smoothed signal')
plt.scatter(range(len(y) - first),y[first:] ,  alpha = 0.3, label = 'Observation')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6), dpi=200)
plt.plot(output_smooth["at"][first:,:].sum(axis=2), label =  'Filtered state')
plt.plot(output_smooth["alpha"][first:,:].sum(axis=2), label = 'Smoothed state')
plt.scatter(range(len(a_array)-first), np.abs(a_array[first:]) ,  alpha = 0.3, label = 'State')
plt.legend()
plt.show()
