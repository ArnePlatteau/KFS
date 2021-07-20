# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 11:33:24 2021

@author: arnep
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ekf import eStateSpacer
import time

t0 = time.time()
np.random.seed(0)
def lin_fun(x, param_vec):
    return  x*param_vec.transpose()

def one_fun(x, param_vec):
    return param_vec

def square_fun(x, param_vec):
    return  (x**2)*param_vec.transpose()

def square_der_fun(x, param_vec):
    return  2*x*param_vec.transpose()


n = 200

Q = np.matrix(4.0)
H=  np.matrix(0.01)
noise_obs  = np.random.normal(0,H,n)
noise_state  = np.random.normal(0, Q,n)

a1 = (0)
a_array = np.zeros((n,1))
a_array[0] = a1
T= np.matrix(0.7)

for i in range(1,n):
    a_array[i]= a_array[i-1]*T.transpose()  + noise_state[i]

Z = np.matrix(1)
y = (a_array**2)*Z.transpose() + noise_obs.reshape(-1,1)

y = np.array(y)
y[50:80] = np.nan
period = np.zeros((25,1))   
period[:] = np.nan
y = np.append(y, period, axis = 0)

a_array[50:80] = np.nan

plt.plot(y)


#create state_space object
Nile_filter = eStateSpacer(y, square_fun, square_der_fun, lin_fun, one_fun )

#set the function
fun = Nile_filter.kalman_llik_diffuse

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
Nile_filter.init_state_matrices( T= T, R=None, Z=None, Q=Q, H= H, 
                            c=None, d=None, states = 1, eta_size = 1)
matr = Nile_filter.init_matr


#estimate MLE parameters
bnds = ((1, 50000),(1, 50000) #,(1, 50000),
   #     (-.999,.999), (-.999,.999),
   #       (d1-5.001,d1 +10.001), (d2-10.001,d2 + 10.001) 
   )
est =  Nile_filter.ml_estimator_matrix(fun, matr, param_loc, filter_init, param_init, bnds)

#get output
at, Pt, a, P, v, F, K, newC, newD = Nile_filter.kalman_filter(Nile_filter.init_matr,filter_init)

at, Pt, a, P, v, F, K, alpha, V, r, N = Nile_filter.smoother(Nile_filter.init_matr,filter_init)

first = 0
#plot data and filtered and smoothed values    
plt.figure(figsize=(10, 6), dpi=200)
plt.plot(at[first:,:].sum(axis=2)**2, label =  'Filtered state')
#plt.plot(alpha[first:,:].sum(axis=2)**2, label = 'Smoothed state')
plt.scatter(range(len(y) - first),y[first:] ,  alpha = 0.3, label = 'Observation')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6), dpi=200)
plt.plot(at[first:,:].sum(axis=2), label =  'Filtered state')
#plt.plot(alpha[first:,:].sum(axis=2), label = 'Smoothed state')
plt.scatter(range(len(a_array)-first), np.abs(a_array[first:]) ,  alpha = 0.3, label = 'State')
plt.legend()
plt.show()

t1 = time.time()

total = t1-t0

print(total)