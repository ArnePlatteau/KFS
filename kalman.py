# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:07:16 2021

@author: arnep
"""
import pandas as pd
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
 

def dim_check(T, R, Z, Q, H, c, d):
    """Returns true if the dimensions are okay """
    
    return True

def collect_3d(dict_syst_matr):
    """
    Takes a dict of system matrices in, and returns a list of the matrix names
    which are in 3D. Used in the Kalman recursions to be able for the functions 
    to work on time-varying system matrices as well as constant system matrices.
    """

    list_3d = list()
    for key in dict_syst_matr.keys():
        if len(dict_syst_matr[key].shape) ==3:
            list_3d.append(key)
    return list_3d

def convert_matrix(*args):
    """
    Convert arrays to matrices
    """

    for el in args:
        el = np.matrix(el)
    return args



class state_spacer():
    def __init__(self, y,*matrices):
        """
        Implementation of the following model:
            yt = c + Zt alphat + epst, epst ~ NID(0,H)
            alphat+1 = d + alphat + Rt etat, etat ~NID(0,Q) 
        
        define time-varying structural matrices in dimension (row, column, time)
        """
        self.y = y
        self.init_state_matrices(*matrices)
        self.init_matrices = True
        

    def init_state_matrices(self, T=None, R=None, Z=None, Q=None, H=None, 
                            c=None, d=None, states = 1, eta_size = 1):
        """ 
        Sets the initial system matrices. When no matrices are specified, default
        initial system matrices are set. It is also possible to detremine the 
        size of the error terms, the default matrices will then be set according
        to the dimensions specified.
        """
        if dim_check(T, R, Z, Q, H, c, d):
            #check to see if the matrices given have valid dimensions 
            
            self.init_matr = {}
            if T is None: 
                self.init_matr['T'] = np.eye((states))
            else:
                self.init_matr['T'] = T
            
            if R is None: 
                self.init_matr['R'] = np.ones((states, eta_size))
            else:
                self.init_matr['R'] = R
                
            if Z is None: 
                self.init_matr['Z'] =  np.ones((self.y.shape[1], states))
            else:
                self.init_matr['Z'] = Z
                
            if Q is None: 
                self.init_matr['Q'] = np.eye(eta_size)
            else:
                self.init_matr['Q'] = Q
                
            if H is None: 
                self.init_matr['H'] = np.eye(self.y.shape[1])
            else:
                self.init_matr['H'] = H

            if c is None: 
                self.init_matr['c'] = np.zeros((self.y.shape[1], 1))
            else:
                self.init_matr['c'] = c

            if d is None: 
                self.init_matr['d'] = np.zeros((states, 1))
            else:
                self.matrices['d'] = d

        else: 
            print("error: dimensions don't match")
            
            
    def get_matrices(self, syst_matr):
        """
        Helper function. It looks which matrices 
        are 3D, collects these in a list, and for all 2D matrices ensures that 
        they are in a np.matrix element.
        """
        matr = {}
        list_3d = collect_3d(syst_matr)

        for el in ['T', 'R', 'Z', 'Q', 'H', 'c', 'd']:
            matr[el] = syst_matr[el]
        
        for el in filter(lambda el: el not in list_3d, matr.keys()):
            matr[el] = np.matrix(matr[el])
            
        return matr, list_3d


    def transit_syst_matrix(self, list_trans, t, matr):
        """
        For the 3D system matrices, the matrix of time t is obtained and put in 
        a np.matrix object.
        """
        for el in list_trans:
            matr[el] = np.matrix(matr[el][:,:,t])
        return matr


    def kalman_filter(self, syst_matr, filter_init):
        """
        Kalman filter recursions, based on the system matrices and the initialisation
        of the filter given. It first gets the processed matrices by calling 
        the helper functions, initialises the output arrays, and then 
        applies the filter by the following equations:
            
        v_t = y_t - Z_t*a_t - c_t
        F_t = Z_t*P_t* Z_t' +  H_t
        K_t = T_t*P_t*Z_t'*F_t-1
        a_{t+1} = T_t* a_t + K_t*v_t + d
        P_{t+1} = T*P_t*T_t' + R_t*Q_t*R_t' - K_t*F_t*K_t' 
        """
        matrices, list_3d = self.get_matrices(syst_matr)
        time = len(self.y)
        a_init = np.matrix(filter_init[0])
        P_init = np.matrix(filter_init[1])

        a    = np.zeros((time + 1, a_init.shape[0], a_init.shape[1]))
        P    = np.zeros((time + 1, P_init.shape[0], P_init.shape[1]))
        F    = np.zeros((time    , self.y.shape[1], self.y.shape[1]))
        K    = np.zeros((time    , a_init.shape[1], self.y.shape[1]))
        v    = np.zeros((time    , self.y.shape[1], 1))
        a[0,:] = a_init
        P[0,:] = P_init

        for t in range(time):
            matr = self.transit_syst_matrix(list_3d, t, matrices.copy())
            T, R, Z, Q, H, c, d = matr['T'], matr['R'],  matr['Z'], matr['Q'], matr['H'], matr['c'], matr['d']   
            yt = y[t]
            #v and a are transposed
            v[t] = yt -a[t]*Z.transpose() - c.transpose() 

            #F, P and K are not transposed
            F[t]   = Z*P[t]*Z.transpose() + H
            K[t]   = T*P[t]*Z.transpose()*np.linalg.inv(F[t])

            a[t+1] = a[t]*T.transpose() + v[t]*K[t].transpose() + d.transpose()
            P[t+1] = T*P[t]*T.transpose() + R*Q*R.transpose() - K[t]*F[t]*K[t].transpose()
 
        return a, P, v, F, K
    
    
    def smoother(self, syst_matr, filter_init):
        """
        Kalman smoothing recursions, based on the system matrices and the initialisation
        of the filter given. It first gets the processed matrices by calling 
        the helper functions, initialises the output arrays. Then, it calls the 
        Kalman filter and uses this to calculate the smoothing recursions. These
        are given by: 
            
        
        r_t = v_t*(F_{t+1}^-1)'*Z_t + r{t+1}*L{t+1}
        N_t = Z'*F_{t+1}^-1*Z_t + L{t+1}*N{t+1}*L{t+1}
        alpha{t+1} = a{t+1} + r[t]*P{t+1}'
        V{t+1} = P{t+1} - P{t+1}*N_t*P{t+1}
        
        """

        matrices, list_3d = self.get_matrices(syst_matr)
        a, P, v, F, K = self.kalman_filter(syst_matr, filter_init)
        
        r = np.zeros((a.shape))
        r[:] = np.nan
        N = np.zeros((P.shape))
        N[:] = np.nan
        r[len(a)-2] = 0
        N[len(a)-2] = 0
        
        alpha = np.zeros(a.shape)
        alpha[:] = np.nan
        V = np.zeros(P.shape)
        V[:] = np.nan
        
        r, N, alpha, V = r[:len(r)-1], N[:len(N)-1], alpha[:len(alpha)-1], V[:len(V)-1]
        for t in range(len(a)-3, -2,-1):
            matr = self.transit_syst_matrix(list_3d, t, matrices.copy())
            Z = matr['Z']
            T = matr['T']
            
            L = T - K[t+1]*Z

            r[t] = v[t+1]*np.linalg.inv(F[t+1]).transpose()*Z + (r[t+1]*L)
            N[t] = Z.transpose()*np.linalg.inv(F[t+1])*Z + L*N[t+1]*L

            alpha[t+1] = a[t+1] + np.dot(r[t],P[t+1].transpose())
            V[t+1] = P[t+1] - P[t+1]*N[t]*P[t+1]
            
        return a, P, v, F, K, alpha, V, r, N
    
    
    def kalman_llik_diffuse(self,param, matr, param_loc, filter_init):
        """
        Diffuse loglikelihood function for the Kalman filter system matrices. 
        The function allows for specification of the elements in the system matrices
        which are optimised, and which are remained fixed. It is not allowed 
        to do maximum likelihood on a time-varying parameter.
        """
        syst_matr = matr
        
        #get the elements which are optimised in the ML function
        for key in param_loc.keys():
            syst_matr[param_loc[key]['matrix']][param_loc[key]['row'],param_loc[key]['col']] = param[key]
        
        print(syst_matr)
        
        #apply Kalman Filter
        a, P, v, F, K  =  self.kalman_filter(syst_matr, filter_init)
        
        #first element not used in diffuse likeilhood
        v = v[1:,:,:]
        F = F[1:,:,:]

        n = len(v)
        accum = 0
        for i in range(n):
            accum += v[i]*np.linalg.inv(F[i])*v[i].transpose()

        #log likelihood function: -n/2 * log(2*pi) - 1/2*sum(log(F_t) + v_t^2/F_t)
        l = -(n / 2) * np.log(2 * np.pi) - (1 / 2) * (np.log(np.linalg.det(F)).sum()) - (1 / 2) * (
                accum)
        llik = -np.mean(l)
        return llik
        
    
    def ml_estimator_matrix(self, fun, matr, param_loc, filter_init, param_init,
                            bnds, method = 'L-BFGS-B',
                            options = {'eps': 1e-07,'disp': True,'maxiter': 200}):
        """ MLE estimator which optimises the likelihood function given, based on 
        initialisation of both the filter and the parameters, bounds, and
        a method """          
        
        #make object with all arguments together              
        args = (matr, param_loc, filter_init)

        #prepare initialisation
        param_init = np.array( list(param_init.values()))
        
        #optimize log_likelihood 
        results = optimize.minimize(fun, param_init,
                                          options=options, args = args,
                                          method=method, bounds=bnds)
        
        #print the parameters and the AIC
        estimates = results.x
        results['llik'] = -results.fun
        results['AIC'] = 2*len(param_init) - results['llik']
        print('params: ' + str(estimates))
        print('likelihood: ' +str( results['llik'] ))
        print('AIC: ' +str(results['AIC']))

        return results

