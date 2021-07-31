# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:07:16 2021

@author: arnep
"""
from scipy import optimize
import numpy as np
 

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
    def __init__(self, *matrices):
        """
        Implementation of the following model:
            yt = c + Zt alphat + epst, epst ~ NID(0,H)
            alphat+1 = d + alphat + Rt etat, etat ~NID(0,Q) 
        
        define time-varying structural matrices in dimension (row, column, time)
        """
        self.init_state_matrices(*matrices)
        self.fitted = False


    def init_state_matrices(self, T=None, R=None, Z=None, Q=None, H=None, 
                            c=None, d=None, y= None, y_dim =1, states = 1, eta_size = 1):
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
                if y is not None: 
                    self.init_matr['Z'] =  np.ones((y.shape[1], states))
                else:
                    self.init_matr['Z'] =  np.ones((y_dim, states))

            else:
                self.init_matr['Z'] = Z
                
            if Q is None: 
                self.init_matr['Q'] = np.eye(eta_size)
            else:
                self.init_matr['Q'] = Q
                
            if H is None: 
                if y is not None: 
                    self.init_matr['H'] = np.eye(y.shape[1])
                else:
                    self.init_matr['H'] = np.eye(y_dim)
                    
            else:
                self.init_matr['H'] = H

            if c is None: 
                if y is not None: 
                    self.init_matr['c'] = np.zeros((y.shape[1], 1))
                else:
                    self.init_matr['c'] = np.zeros((y_dim, 1))

            else:
                self.init_matr['c'] = c

            if d is None: 
                self.init_matr['d'] = np.zeros((states, 1))
            else:
                self.init_matr['d'] = d

        
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


    def kalman_init(self,y, filter_init, time):
        """Helper function, which defines all the necessary output matrices and 
        initialises."""
        
        a_init = np.matrix(filter_init[0])
        P_init = np.matrix(filter_init[1])
    
        at   = np.zeros((time, a_init.shape[0], a_init.shape[1]))
        Pt   = np.zeros((time, P_init.shape[0], P_init.shape[1]))
        a    = np.zeros((time + 1, a_init.shape[0], a_init.shape[1]))
        P    = np.zeros((time + 1, P_init.shape[0], P_init.shape[1]))
        F    = np.zeros((time    , y.shape[1], y.shape[1]))
        K    = np.zeros((time    , a_init.shape[1], y.shape[1]))
        v    = np.zeros((time    , y.shape[1], 1))
        a[0,:] = a_init
        P[0,:] = P_init
        return at, Pt, a, P, F, K, v
    
    
    def kalman_filter_iteration(self, yt, a, P, Z, T, c, d, H, Q, R, 
                                v, F, att, Ptt ):
        """
        Normal Kalman iteration
        
        v_t = y_t - Z_t*a_t - c_t
        F_t = Z_t*P_t* Z_t' +  H_t
        K_t = T_t*P_t*Z_t'*F_t-1
        a_{t+1} = T_t* a_t + K_t*v_t + d
        P_{t+1} = T*P_t*T_t' + R_t*Q_t*R_t' - K_t*F_t*K_t' 
        """
        
        #v and a are transposed
        v = yt -a*Z.transpose() - c.transpose() 
    
        #F, P and K are not transposed
        F = Z*P*Z.transpose() + H
        M = P*Z.transpose()*np.linalg.inv(F)
        K = T*M
        
        att = a + v*M.transpose()
        Ptt = P - M.transpose()*P*M
        
        at1 = a*T.transpose() + v*K.transpose() + d.transpose()
        Pt1 = T*P*T.transpose() + R*Q*R.transpose() - K*F*K.transpose()
        
        return v, F, K, att, Ptt, at1, Pt1, c, d


    def kalman_filter_iteration_missing(self, yt, a, P, Z, T, c, d, H, Q, R,
                                v, F, att, Ptt, tol = 1e7 ):
        """
        Kalman iteration function in case the observation is missing.
        
        v_t = undefined
        F_t = infinity
        K_t = 0
        a_{t+1} = T_t* a_t + d
        P_{t+1} = T*P_t*T_t' + R_t*Q_t*R_t'
        """
        
        #v and a are transposed
        v = yt
    
        #F, P and K are not transposed
        F = np.matrix(np.ones(H.shape)*tol)
        M = np.matrix(np.zeros((P*Z.transpose()*np.linalg.inv(F)).shape))
        K = T*M
        
        att = a 
        Ptt = P
        
        at1 = a*T.transpose() + d.transpose()
        Pt1 = T*P*T.transpose() + R*Q*R.transpose() 
        
        return v, F, K, att, Ptt, at1, Pt1, c, d


    def create_empty_objs(self, yt, H, at, Pt):
        """Helper function to create certain empty objects, which are later 
        used in the code.
        """
        v_obj = np.zeros(yt.shape)
        F_obj = np.zeros(H.shape)
        att_obj = np.zeros(at.shape)
        Ptt_obj = np.zeros(Pt.shape)
        return v_obj, F_obj, att_obj, Ptt_obj


    def kalman_filter(self, y, filter_init, syst_matr = None):
        """
        Kalman filter recursions, based on the system matrices and the initialisation
        of the filter given. It first gets the processed matrices by calling 
        the helper functions, initialises the output arrays, and then 
        applies the filter.
        """
            
        time = len(y)
        if syst_matr is None:
            syst_matr = self.init_matr
        
        matrices, list_3d = self.get_matrices(syst_matr)
        
        at, Pt, a, P, F, K, v = self.kalman_init(y, filter_init, time)
        
        t = 0
        T, R, Z, Q, H, c, d = self.get_syst_matrices(list_3d, t, matrices)
        
        yt = np.zeros(y[t].shape)   
        
        newC = np.zeros((self.init_matr['c'].shape[0], self.init_matr['c'].shape[1], time))
        newD = np.zeros((self.init_matr['d'].shape[0], self.init_matr['c'].shape[1], time ))

        v_obj, F_obj, att_obj, Ptt_obj = self.create_empty_objs(yt, H, a[t], P[t])
                
        if np.isnan(np.sum(y)):
            for t in range(time):
                T, R, Z, Q, H, c, d = self.get_syst_matrices(list_3d, t, matrices)
                yt = y[t]
                
                if not np.isnan(yt):
                    v[t], F[t], K[t], at[t], Pt[t], a[t+1], P[t+1], newC[:,:,t], newD[:,:,t] = self.kalman_filter_iteration(yt, a[t], P[t], Z, T, c, d, H, Q, R, 
                                                                                                                    v_obj, F_obj, att_obj, Ptt_obj )
                else:
                    v[t], F[t], K[t], at[t], Pt[t], a[t+1], P[t+1], newC[:,:,t], newD[:,:,t] = self.kalman_filter_iteration_missing(yt, a[t], P[t], Z, T, c, d, H, Q, R, 
                                                                                                                                v_obj, F_obj, att_obj, Ptt_obj )
  
        else: 
            for t in range(time):
                T, R, Z, Q, H, c, d = self.get_syst_matrices(list_3d, t, matrices)
                yt = y[t]
                
                v[t], F[t], K[t], at[t], Pt[t], a[t+1], P[t+1], newC[:,:,t], newD[:,:,t] = self.kalman_filter_iteration(yt, a[t], P[t], Z, T, c, d, H, Q, R, 
                                                                                                                    v_obj, F_obj, att_obj, Ptt_obj )
                                                                                                                            
        return at, Pt, a, P, v, F, K, newC, newD
    
    
    def smoothing_iteration(self, v, F, r, T, K, Z, N, P, a):
        L = T - K*Z
        r= v*np.linalg.inv(F).transpose()*Z + (r*L)
        N = Z.transpose()*np.linalg.inv(F)*Z + L*N*L
        alpha = a + np.dot(r,P.transpose())
        V = P - P*N*P
        return L, r, N, alpha, V


    def smoothing_iteration_missing(self, v, F, r, T, K, Z, N, P, a):    
        L = T 
        r= r*L
        N = L*N*L
        alpha = a + np.dot(r,P.transpose())
        V = P - P*N*P
        return L, r, N, alpha, V


    def get_syst_matrices(self, list_3d, t, matrices):
        matr = self.transit_syst_matrix(list_3d, t, matrices.copy())
        T, R, Z, Q, H, c, d = matr['T'], matr['R'],  matr['Z'], matr['Q'], matr['H'], matr['c'], matr['d']   
        return T, R, Z, Q, H, c, d
    
        
    def smoother(self, y, filter_init):
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
        
        matrices, list_3d = self.get_matrices(self.init_matr)
        at, Pt, a, P, v, F, K, newC, newD =self.kalman_filter(y, filter_init)
        

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
        if np.isnan(np.sum(v)):
            for t in range(len(a)-3, -1,-1):
                T, _, Z, _, _, _, _ = self.get_syst_matrices(list_3d, t, matrices.copy())
                
                if not np.isnan(v[t+1]):
                    L, r[t], N[t], alpha[t+1], V[t+1] = self.smoothing_iteration(v[t+1], F[t+1], r[t+1], T, K[t+1], Z, N[t+1], P[t+1], a[t+1])
                else: 
                    L, r[t], N[t], alpha[t+1], V[t+1] = self.smoothing_iteration_missing(v[t+1], F[t+1], r[t+1], T, K[t+1], Z, N[t+1], P[t+1], a[t+1])
               
            t = - 1
            T, _, Z, _, _, _, _ = self.get_syst_matrices(list_3d, t, matrices.copy())
    
            if not np.isnan(v[t+1]):
                _, _, _, alpha[t+1], V[t+1] = self.smoothing_iteration(v[t+1], F[t+1], r[t+1], T, K[t+1], Z, N[t+1], P[t+1], a[t+1])
            else: 
                _, _, _, alpha[t+1], V[t+1] = self.smoothing_iteration_missing(v[t+1], F[t+1], r[t+1], T, K[t+1], Z, N[t+1], P[t+1], a[t+1])
   
        else: 
            for t in range(len(a)-3, -1,-1):
                T, _, Z, _, _, _, _ = self.get_syst_matrices(list_3d, t, matrices.copy())
                
                L, r[t], N[t], alpha[t+1], V[t+1] = self.smoothing_iteration(v[t+1], F[t+1], r[t+1], T, K[t+1], Z, N[t+1], P[t+1], a[t+1])

            t = - 1
            T, _, Z, _, _, _, _ = self.get_syst_matrices(list_3d, t, matrices.copy())
                
            _, _, _, alpha[t+1], V[t+1] = self.smoothing_iteration(v[t+1], F[t+1], r[t+1], T, K[t+1], Z, N[t+1], P[t+1], a[t+1])


        return at, Pt, a, P, v, F, K, alpha, V, r, N
    
    
    def kalman_llik_diffuse(self,param, y, matr, param_loc, filter_init):
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

        #apply Kalman Filter
        at, Pt, a, P, v, F, K, newC, newD  =  self.kalman_filter(y, filter_init, syst_matr)
        
        #first element not used in diffuse likeilhood
        v = v[1:,:,:]
        F = F[1:,:,:]
        
        v_temp = v.copy()
        v_temp[np.isnan(v_temp)] = 0
        
        n = len(v)
        accum = 0
        for i in range(n):
            accum += v_temp[i]*np.linalg.inv(F[i])*v_temp[i].transpose()

        #log likelihood function: -n/2 * log(2*pi) - 1/2*sum(log(F_t) + v_t^2/F_t)
        l = -(n / 2) * np.log(2 * np.pi) - (1 / 2) * (np.log(np.linalg.det(F)).sum()) - (1 / 2) * (
                accum)
        llik = -np.mean(l)
        return llik
        
    
    def ml_estimator_matrix(self, y, fun, matr, param_loc, filter_init, param_init,
                            bnds, method = 'L-BFGS-B',
                            options = {'eps': 1e-07,'disp': True,'maxiter': 200}):
        """ MLE estimator which optimises the likelihood function given, based on 
        initialisation of both the filter and the parameters, bounds, and
        a method """          
        
        #make object with all arguments together              
        args = (y, matr, param_loc, filter_init)

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
    
    
    def fit(self, y, fun, param_loc, filter_init, param_init,
                  bnds, method = 'L-BFGS-B',
                  options = {'eps': 1e-07,'disp': True,'maxiter': 200}):
        
        syst_matr = self.init_matr
        
        res = self.ml_estimator_matrix( y, fun, syst_matr, param_loc,
                                       filter_init, param_init,
                            bnds, method = method, options = options)
        
        
        param = res.x

        #get the elements which are optimised in the ML function
        for key in param_loc.keys():
            syst_matr[param_loc[key]['matrix']][param_loc[key]['row'],param_loc[key]['col']] = param[key]
        
        for key in self.init_matr.keys():
            self.init_matr[key] = syst_matr[key]
            
        self.fitted = True

        return self

"""    
    def disturbance_smoothing_errors(self, ):
        # initialise dataframe
        v, F, K, r, N = self.res['v'], self.res['F'],self.res['K'], self.res['r'], self.res['N']
        
        # calculate u = v_t/F_t - K_t*r_t
        u = v * np.linalg.inv(F) - K * r
    
        # calculate D_t = 1/F_t + K_t^2 * N_t
        D = 1 * np.linalg.inv(F) + (K ** 2) * N
    
        # estimated epsilon_t= sigma2_epsilon * u_t
        epsilon_hat = H  u
    
        # estimated conditional variance_t epsilon = sigma2_epsilon - D_t *sigma2_epsilon^2
        var_epsilon_cond = sigma2_epsilon - temp['D'] * (sigma2_epsilon ** 2)
    
        # estimated eta_t= sigma2_eta * r_t
        eta_hat = temp['r'] * sigma2_eta
    
        # estimated conditional variance_t eta = sigma2_eta - N_t *sigma2_eta^2
        var_eta_cond = sigma2_eta - temp['N'] * (sigma2_eta ** 2)
    
        self.diagnostic['u'] = u
        self.diagnostic['D'] = D
        self.diagnostic['epsilon_hat'] = epsilon_hat
        self.diagnostic['var_epsilon_cond'] = var_epsilon_cond
        self.diagnostic['eta_hat'] = eta_hat
        self.diagnostic['var_eta_cond'] = var_eta_cond
"""