# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:07:16 2021

@author: arnep
"""


from scipy import optimize
import numpy as np
import json
from scipy.stats import norm

 
def llik_gaussian(v, F):
    """
    Log likelihood function of a Gaussian model, based on the prediction error 
    decomposition.

    Parameters
    ----------
    v : array-like
        Prediction errors. 1D or 2D.
    F : array-like
        Prediction error variance. 2D or 3D array. 

    Returns
    -------
    llik : integer
        Negative loglikeilhood.

    """
    #set all elements which are nan to zero
    v_temp = v.copy()
    v_temp[np.isnan(v_temp)] = 0
    
    #get length of the error terms
    T = len(v)
    
    #compute the sum of vt*Ft^-1*vt' for all t in T
    accum = 0
    for t in range(T):
        accum += v_temp[t]*np.linalg.inv(F[t])*v_temp[t].transpose()

    #log likelihood function: -n/2 * log(2*pi) - 1/2*sum(log(F_t) + v_t^2/F_t)
    l = -(T / 2) * np.log(2 * np.pi) - (1 / 2) * (np.log(np.linalg.det(F)).sum()) - (1 / 2) * (
            accum)
    
    #use of mean (sum would also be possible, but in the optimisation this would amount to the same)
    llik = -np.mean(l) #negative as minimizer function is used
    return llik


def ml_estimator_matrix( y, matr, param_loc, kalman_llik, filter_init, param_init,
                            bnds,  method = 'L-BFGS-B',
                            options = {'eps': 1e-07,'disp': True,'maxiter': 200}, **llik_kwargs):
        """
        MLE estimator which optimises the likelihood function given, based on 
        initialisation of both the filter and the parameters, bounds, and
        a method.
        
        Parameters
        ----------
        y : array-like 
            Observations used for the fitting of a model.
            matr : dict. System matrices of the state space model.
        param_loc : dict
            Locations of the parameters to be optimised in their 
            respective matrices.
        kalman_llik : function. 
            Log likelihood function to be optimised.
        filter_init : tuple
            Initialisation of the Kalman filter.
        param_init : tuple
            Initialisation of the parameters.
        bnds : tuple
            Bounds for the parameters in the optimisation.
        method : string, optional
            method used for the optimisation of the likelihood function. 
            The default is 'L-BFGS-B'.
        options : dict, optional
            Options for the optimize.minimize function. 
            The default is {'eps': 1e-07,'disp': True,'maxiter': 200}.
        **llik_kwargs : further arguments for the log likelihood function.
    
        Returns
        -------
        results : dict
            Output of optimize.minimize function

        """
        
        #make object with all arguments together              
        args = (y, matr, param_loc, filter_init, llik_kwargs)

        #optimize log_likelihood 
        results = optimize.minimize(kalman_llik, param_init,
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

### implement fully!!!!
def dim_check(T, R, Z, Q, H, c, d):
    """
    Returns true if the dimensions are okay 
    Parameters
    ----------
    T : array-like
        System matrix T.
    R : array-like
        System matrix R.
    Z : array-like
        System matrix Z.
    Q : array-like
        System matrix Q.
    H : array-like
        System matrix H.
    c : array-like
        System matrix c.
    d : array-like
        System matrix d.

    Returns
    -------
    bool
        returns True if dimensions are appropriate and False otherwise.

    """
    
    return True

def collect_3d(dict_syst_matr):
    """
    Takes a dict of system matrices in, and returns a list of the matrix names
    which are in 3D. Used in the Kalman recursions to be able for the functions 
    to work on time-varying system matrices as well as constant system matrices.

    Parameters
    ----------
    dict_syst_matr : dict
        dict with all system matrices.

    Returns
    -------
    list_3d : list
        list of system matrices in 3D.

    """
    
    list_3d = list()
    for key in dict_syst_matr.keys():
        if len(dict_syst_matr[key].shape) >2:
            list_3d.append(key)
    return list_3d


def convert_matrix(*args):
    """
    Convert arrays to matrices

    Parameters
    ----------
    *args : list
            arrays to be converted to matrices.

    Returns
    -------
    args : np.matrix
           Arrays converted in matrices.

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

        Parameters
        ----------
        *matrices : dict
                    System matrices of the state space model.

        Returns
        -------
        None.

        """
        
        self.init_matrices(*matrices)
        self.fit_parameters = {}
        self.fit_results = {}
        self.fitted = False


    def init_matrices(self, T=None, R=None, Z=None, Q=None, H=None, 
                            c=None, d=None, y= None, y_dim =1, states = 1, eta_size = 1):
        """
        Sets the initial system matrices. When no matrices are specified, default
        initial system matrices are set. It is also possible to detremine the 
        size of the error terms, the default matrices will then be set according
        to the dimensions specified.

        Parameters
        ----------
        T : array-like, optional
            T system matrix. Can be 1D, 2D or 3D. If not filled in, T is an 
            np.eye matrix with the dimensions equalling the number of states.
            The default is None.
        R : array-like, optional
            R system matrix. Can be 1D, 2D or 3D.  If not filled in, R is a 
            matrix with ones with the dimensions the states and the number of 
            eta error terms.
            The default is None.
        Z : array-like, optional
            Z system matrix. Can be 1D, 2D or 3D. If not  filled in, Z is a 
            matrix with ones with the dimensions the number of time series in y
            and the number of states. 
            The default is None.
        Q : array-like, optional
            Q system matrix. Can be 1D, 2D or 3D. If not filled in, Q is an 
            eye matrix with the dimensions the number of eta error terms.
            The default is None.
        H : array-like, optional
            H system matrix. Can be 1D, 2D or 3D. If not filled in, H is an 
            eye matrix with the dimensions the number of epsilon error terms.
            The default is None.
        c : array-like, optional
            c system matrix. Can be 1D, 2D or 3D. If not filled in, c is an 
            vector with the dimensions the number of time series in y.
            The default is None.
        d : array-like, optional
            d system matrix. Can be 1D, 2D or 3D. If not filled in, d is a 
            vector with the dimensions the number of states.
            The default is None.
        y : array-like, optional
            Data. When added, this allows the function to correctly specify
            the system matrices dimensions. Specifying explicit matrices
            may (partially) override the information provided here. 
            The default is None.
        y_dim : integer, optional
            Number of time series in y. Instead of adding the data, this number
            can be added, which allows the function to correctly specify the 
            system matrix dimensions. Specifying explicit matrices
            may (partially) override the information provided here.
            The default is 1.
        states : integer, optional
            Number of states desired in the state space model. Specifying
            explicit matrices may (partially) override the information provided 
            here.The default is 1.
        eta_size : integer optional
            number of eta terms to be added in the state space model. Specifying 
            explicit matrices may (partially) override the information provided 
            here.
            The default is 1.

        Returns
        -------
        None.

        """
        
        #check to see if the matrices given have valid dimensions 
        if dim_check(T, R, Z, Q, H, c, d):
            
            self.matr = {}
            if T is None: 
                self.matr['T'] = np.eye((states))
            else:
                self.matr['T'] = np.array(T)
            self.matr['T'] = self.matr['T'].astype(float)
            
            if R is None: 
                self.matr['R'] = np.ones((states, eta_size))
            else:
                self.matr['R'] = np.array(R)
            self.matr['R'] = self.matr['R'].astype(float)
            
            if Z is None: 
                if y is not None: 
                    self.matr['Z'] =  np.ones((y.shape[1], states))
                else:
                    self.matr['Z'] =  np.ones((y_dim, states))
            else:
                self.matr['Z'] = np.array(Z)
            self.matr['Z'] = self.matr['Z'].astype(float)
                
            if Q is None: 
                self.matr['Q'] = np.eye(eta_size)
            else:
                self.matr['Q'] = np.array(Q)
            self.matr['Q'] = self.matr['Q'].astype(float)
                
            if H is None: 
                if y is not None: 
                    self.matr['H'] = np.eye(y.shape[1])
                else:
                    self.matr['H'] = np.eye(y_dim)
            else:
                self.matr['H'] = np.array(H)
            self.matr['H'] = self.matr['H'].astype(float)

            if c is None: 
                if y is not None: 
                    self.matr['c'] = np.zeros((y.shape[1], 1))
                else:
                    self.matr['c'] = np.zeros((y_dim, 1))
            else:
                self.matr['c'] = np.array(c)
            self.matr['c'] = self.matr['c'].astype(float)
            
            if d is None: 
                self.matr['d'] = np.zeros((self.matr['T'].shape[0],1))
            else:
                self.matr['d'] = np.array(d)
            self.matr['d'] = self.matr['d'].astype(float)
        
            self.list_3d = collect_3d(self.matr)
        else: 
            print("error: dimensions don't match")
            
            
    def get_matrices(self, syst_matr):
        """
        Helper function. It looks which matrices 
        are 3D, collects these in a list, and for all 2D matrices ensures that 
        they are in a np.matrix element.    

        Parameters
        ----------
        syst_matr : dict
            Dict containing the system matrices.

        Returns
        -------
        syst_matr : dict
            Dict where all 2D matrices are in a np.matrix() object.
        list_3d : list
            list of 3D matrices.

        """
        
        #get list of the matrices in 3D
        list_3d = collect_3d(syst_matr)
 
        #ensure the 2D matrices are in a np.matrix() object
        for el in filter(lambda el: el not in list_3d, syst_matr.keys()):
            syst_matr[el] = np.matrix(syst_matr[el])
            
        return syst_matr, list_3d


    def get_syst_matrices(self, list_3d, t, matrices):
        """
        Function which unpacks the dict with all the system matrices for time 
        t so that this can be conveniently used in the recursions.
        
        Parameters
        ----------
        list_3d : list
            List of matrices which are in 3D.
        t : integer
            time t.
        matrices : dict
            Dict of matrices.

        Returns
        -------
        T : np.matrix()
            System matrix Tt.
        R : np.matrix()
            System matrix Rt.
        Z : np.matrix()
            System matrix Zt.
        Q : np.matrix()
            System matrix Qt.
        H : np.matrix()
            System matrix Ht.
        c : np.matrix()
            System matrix ct.
        d : np.matrix()
            System matrix dt.

        """
        
        #get the dict with the sytem matrices of time t
        matr = self.transit_syst_matrix(list_3d, t, matrices.copy())

        #return this unpacked
        return matr['T'], matr['R'],  matr['Z'], matr['Q'], matr['H'], matr['c'], matr['d']   


    def transit_syst_matrix(self, list_trans, t, matr):
        """
        For the 3D system matrices, the matrix of time t is obtained and put in 
        a np.matrix object.

        Parameters
        ----------
        list_trans : list
            List of transition matrices which are in 3D.
        t : integer
            Time t, for the system matrices in 3D.
        matr : dict
            System matrices (where some are 3D).

        Returns
        -------
        matr : dict
            System matrices, where the relevant 2D matrix is chosen.

        """
        
        for el in list_trans:
            matr[el] = np.matrix(matr[el][:,:,t])
        return matr


    def kalman_init(self,y, filter_init, time):
        """
        Helper function, which defines all the necessary output matrices and 
        initialises.

        Parameters
        ----------
        y : array-like
            Observations data.
        filter_init : tuple
            Initialisation of the Kalman filter.
        time : integer
            number of Kalman iterations to be done.

        Returns
        -------
        at : array-like
            empty array for at.
        Pt : array-like
            empty array for Pt.
        a : array-like
            empty array for a.
        P : array-like
            empty array for P.
        F : array-like
            empty array for F.
        K : array-like
            empty array for K.
        v : array-like
            empty array for v.

        """
        
        #get initialisation of the filter
        a_init = np.matrix(filter_init[0])
        P_init = np.matrix(filter_init[1])
    
        #create empty arrays
        at   = np.zeros((time, a_init.shape[0], a_init.shape[1]))
        Pt   = np.zeros((time, P_init.shape[0], P_init.shape[1]))
        a    = np.zeros((time + 1, a_init.shape[0], a_init.shape[1]))
        P    = np.zeros((time + 1, P_init.shape[0], P_init.shape[1]))
        F    = np.zeros((time    , y.shape[1], y.shape[1]))
        K    = np.zeros((time    , a_init.shape[1], y.shape[1]))
        v    = np.zeros((time    , y.shape[1], 1))
        
        #fill first element with the initialisation
        a[0,:] = a_init
        P[0,:] = P_init
        
        return at, Pt, a, P, F, K, v
    
    
    def kalman_filter_iteration(self, yt, a, P, Z, T, c, d, H, Q, R, 
                                v, F, att, Ptt ):
        """
        Single iteration of the Kalman filter.         
        v_t = y_t - Z_t*a_t - c_t
        F_t = Z_t*P_t* Z_t' +  H_t
        K_t = T_t*P_t*Z_t'*F_t-1
        a_{t+1} = T_t* a_t + K_t*v_t + d
        P_{t+1} = T*P_t*T_t' + R_t*Q_t*R_t' - K_t*F_t*K_t' 


        Parameters
        ----------
        yt : int or array-like
            Observation data at time t.
        a : int or array-like
            State prediction for time t.
        P : int or array-like
            Variance of state prediction for time t.
        Z : array-like
            System matrix Zt.
        T : array-like
            System matrix Tt.
        c : array-like
            System matrix ct.
        d : array-like
            System matrix dt.
        H : array-like
            System matrix Ht.
        Q : array-like
            System matrix Qt.
        R : array-like
            System matrix Rt.
        v : int or array-like
            Previous prediction error.
        F : int or array-like
            Previous prediction error variance.
        att : int or array-like
            Previous filtered state (t-1).
        Ptt : int or array-like
            Previous filtered state variance (t-1).

        Returns
        -------
        v : int or array-like
            New prediction error.
        F : int or array-like
            New prediction error variance.
        K : int or array-like
            New K.
        att : int or array-like
            New filtered state (time t).
        Ptt : int or array-like
            New filtered state variance (time t).
        at1 : int or array-like
            New state prediction for t + 1.
        Pt1 : int or array-like
            Variance of state prediction for t + 1.
        c : array-like
            Just c, no transformation happens in normal Kalman filter.
        d : array-like
            Just d, no transformation happens in normal Kalman filter.
        """
        
        #v and a are transposed
        v = yt -a*Z.transpose() - c.transpose() 
    
        #F, P and K are not transposed
        F = Z*P*Z.transpose() + H
        M = P*Z.transpose()*np.linalg.inv(F)
        K = T*M
        
        att = a + v*M.transpose()
        Ptt = P - M*F*M.transpose()
        
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

        Parameters
        ----------
        yt : int or array-like
            Observation data at time t.
        a : int or array-like
            State prediction for time t.
        P : int or array-like
            Variance of state prediction for time t.
        Z : array-like
            System matrix Zt.
        T : array-like
            System matrix Tt.
        c : array-like
            System matrix ct.
        d : array-like
            System matrix dt.
        H : array-like
            System matrix Ht.
        Q : array-like
            System matrix Qt.
        R : array-like
            System matrix Rt.
        v : int or array-like
            Previous prediction error. (not used in the code, placeholder)
        F : int or array-like
            Previous prediction error variance. (not used in the code, placeholder)
        att : int or array-like
            Previous filtered state (t-1).
        Ptt : int or array-like
            Previous filtered state variance (t-1).
        tol : int or float, optional
            High value which in theory should go to infinity. The default is 1e7.

        Returns
        -------
        v : int or array-like
            New prediction error.
        F : int or array-like
            New prediction error variance.
        K : int or array-like
            New K.
        att : int or array-like
            New filtered state (time t).
        Ptt : int or array-like
            New filtered state variance (time t).
        at1 : int or array-like
            New state prediction for t + 1.
        Pt1 : int or array-like
            Variance of state prediction for t + 1.
        c : array-like
            Just c, no transformation happens in normal Kalman filter.
        d : array-like
            Just d, no transformation happens in normal Kalman filter.

        """
        
        #v and a are transposed
        v = yt*np.nan
    
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
        """
        Helper function to create certain empty objects, which are later 
        used in the code.
        
        Parameters
        ----------
        yt : array-like
            Observations.
        H : array-like
            H system matrix of time t.
        at : array-like
            array in the form of the filtered state.
        Pt : array-like
            array in the form of the filtered state variance.

        Returns
        -------
        v_obj : array-like
            empty v array.
        F_obj : array-like
            empty F array.
        att_obj : array-like
            empty att array.
        Ptt_obj : array-like
            empty Ptt array.

        """

        v_obj = np.zeros(yt.shape)
        F_obj = np.zeros(H.shape)
        att_obj = np.zeros(at.shape)
        Ptt_obj = np.zeros(Pt.shape)
        return v_obj, F_obj, att_obj, Ptt_obj


    def kalman_filter_base(self, y, filter_init, syst_matr):
        """
        Kalman filter recursions, based on the system matrices and the initialisation
        of the filter given. It first gets the processed matrices by calling 
        the helper functions, initialises the output arrays, and then 
        applies the filter.

        Parameters
        ----------
        y : array-like
            Observation data.
        filter_init : tuple
            Initialisation of the filter.
        syst_matr : dict
            Dictionnary containging all the system matrices.

        Returns
        -------
        at : array-like
            Filtered states.
        Pt : array-like
            Filtered state variances.
        a : array-like
            Filtered state predictions.
        P : array-like
            Filtered state prediction variances.
        v : array-like
            Filtered prediction errors.
        F : array-like
            Filtered prediction error variances.
        K : array-like
            Filtered K (convenient result for other computations).
        newC : array-like
            same as c in the normal Kalman filter, given for coherence with other 
            methods.
        newD : array-like
            same as d in the normal Kalman filter, given for coherence with other 
            methods.
        """
                 
        #get the length of the array
        time = len(y)

        #convert system arrays to matrices, and get the 3D system matrices
        matrices, list_3d = self.get_matrices(syst_matr)
        
        #initialise the Kalman filter
        at, Pt, a, P, F, K, v = self.kalman_init(y, filter_init, time)
        
        #get the system matrices belonging to the first observation
        t = 0
        T, R, Z, Q, H, c, d = self.get_syst_matrices(list_3d, t, matrices)
        
        #get an array in the shape of the first observation
        yt = np.zeros(y[t].shape)   
        
        #initialise the arrays for new c and new d. Not used in the base filter
        #only here for compability with more advanced filters
        newC = np.zeros((self.matr['c'].shape[0], self.matr['c'].shape[1], time))
        newD = np.zeros((self.matr['d'].shape[0], self.matr['c'].shape[1], time ))

        #create empty objects for the results
        v_obj, F_obj, att_obj, Ptt_obj = self.create_empty_objs(yt, H, a[t], P[t])
                
        #check if there is a nan in the observations
        if np.isnan(np.sum(y)):
            #go over the observation array
            for t in range(time):
                #get system matrices and the observation at time t
                T, R, Z, Q, H, c, d = self.get_syst_matrices(list_3d, t, matrices)
                yt = y[t]
                args = yt, a[t], P[t], Z, T, c, d, H, Q, R, v_obj, F_obj, att_obj, Ptt_obj
                #in case the observation is not missing: base iteration
                if not np.isnan(yt):
                    v[t], F[t], K[t], at[t], Pt[t], a[t+1], P[t+1], newC[:,:,t], newD[:,:,t] = self.kalman_filter_iteration(*args)
                #else, the missing observation iteration is performed
                else:
                    v[t], F[t], K[t], at[t], Pt[t], a[t+1], P[t+1], newC[:,:,t], newD[:,:,t] = self.kalman_filter_iteration_missing(*args )
  
        #this is the workflow if no observations are missing 
        else: 
            #go over the observation array
            for t in range(time):
                #get system matrices and the observation at time t
                T, R, Z, Q, H, c, d = self.get_syst_matrices(list_3d, t, matrices)
                yt = y[t]
                
                args = yt, a[t], P[t], Z, T, c, d, H, Q, R, v_obj, F_obj, att_obj, Ptt_obj
                #perform base iteration
                v[t], F[t], K[t], at[t], Pt[t], a[t+1], P[t+1], newC[:,:,t], newD[:,:,t] = self.kalman_filter_iteration(*args)
                                                                                                                            
        return at, Pt, a, P, v, F, K, newC, newD
    
    
    def kalman_filter(self, y, filter_init):
        """
        Function which executes the Kalman filter base, and stores the results 
        in a dict, which is more convenient for final users.

        Parameters
        ----------
        y : array-like
            Observation data.
        filter_init : tuple
            Initalisation of the filter.

        Returns
        -------
        o : Dict
            Filter output.

        """
        o = {}
        o["at"], o["Pt"], o["a"], o["P"], o["v"], o["F"], o["K"], o["newC"], o["newD"]  =  self.kalman_filter_base(y, filter_init, self.matr)
        return o
        
    
    def smoothing_iteration(self, v, F, r, T, K, Z, N, P, a):
        """
        Single smoothing iteration recursion when the observation is available.
        Lt+1 = Tt+1 - Kt+1 Zt+1
        r_t = v_t+1*(F_{t+1}^-1)'*Z_t + r{t+1}*L{t+1}
        N_t = Z'*F_{t+1}^-1*Z_t + L{t+1}*N{t+1}*L{t+1}
        alpha{t+1} = a{t+1} + r[t]*P{t+1}'
        V{t+1} = P{t+1} - P{t+1}*N_t*P{t+1}

        Parameters
        ----------
        v : array-like
            Prediction error (value of t+1).
        F : array-like
            Prediction error variance (value of t+1).
        r : array-like
            Intermediate result in the smoothing recursions (value of t+1).
        T : array-like
            System matrix T (value of t+1).
        K : array-like
            Intermediate result K in the filter recursions (value of t+1).
        Z : array-like
            System matrix Z (value of t+1).
        N : array-like
            Intermediate result N in the filter recursions (value of t+1).
        P : array-like
            State prediction variance (value of t+1).
        a : array-like
            State prediction (value of t+1).

        Returns
        -------
        L : array-like
            Intermediate result in the smoothing recursions (value of t).
        r : array-like
            Intermediate result in the smoothing recursions (value of t).
        N : array-like
            Intermediate result N in the filter recursions (value of t).
        alpha : array-like
            Smoothed state (value of t).
        V : array-like
            Smoothed state variance (value of t).

        """        
        
        L = T - K*Z
        r= v*np.linalg.inv(F).transpose()*Z + (r*L)
        N = Z.transpose()*np.linalg.inv(F)*Z + L.transpose()*N*L
        alpha = a + np.dot(r,P.transpose())
        V = P - P*N*P
        return L, r, N, alpha, V


    def smoothing_iteration_missing(self, v, F, r, T, K, Z, N, P, a):    
        """
        Single smoothing iteration recursion when the observation is missing.
        Lt+1 = Tt+1- Kt+1*Zt+1
        r_t =  r{t+1}*L{t+1}
        N_t = L{t+1}*N{t+1}*L{t+1}
        alpha{t+1} = a{t+1} + r[t]*P{t+1}'
        V{t+1} = P{t+1} - P{t+1}*N_t*P{t+1}

        Parameters
        ----------
        v : array-like
            Prediction error (value of t+1).
        F : array-like
            Prediction error variance (value of t+1).
        r : array-like
            Intermediate result in the smoothing recursions (value of t+1).
        T : array-like
            System matrix T (value of t).
        K : array-like
            Intermediate result K in the filter recursions (value of t+1).
        Z : array-like
            System matrix Z (value of t).
        N : array-like
            Intermediate result N in the filter recursions (value of t+1).
        P : array-like
            State prediction variance (value of t+1).
        a : array-like
            State prediction (value of t+1).
            
        Returns
        -------
        L : array-like
            Intermediate result in the smoothing recursions (value of t).
        r : array-like
            Intermediate result in the smoothing recursions (value of t).
        N : array-like
            Intermediate result N in the filter recursions (value of t).
        alpha : array-like
            Smoothed state (value of t).
        V : array-like
            Smoothed state variance (value of t).

        """
        
        L = T 
        r= r*L
        N = L.transpose()*N*L
        alpha = a + np.dot(r,P.transpose())
        V = P - P*N*P
        
        return L, r, N, alpha, V
    
        
    def smoother_base(self, y, filter_init, return_smoothed_errors=True):
        """
        Kalman smoothing recursions, based on the system matrices and the 
        initialisation of the filter given. It first gets the processed matrices 
        by calling the helper functions, initialises the output arrays. Then, 
        it calls the  Kalman filter and uses this to calculate the smoothing 
        recursions in separate functions, depending on whether the observation
        is missing or not. The smoothed errors are also computed if indicated.

        Parameters
        ----------
        y : array-like
            Observation data.
        filter_init : tuple
            Initialisation of the Kalman filter.
        return_smoothed_errors : boolean, optional
            Indicates whether the smoothed errors also should be computed. 
            The default is True.

        Returns
        -------
        Kalman smoothing output : several array-like items
            Output of the Kalman smoother (at, Pt, a, P, v, F, K, newC, newD
            the same as the output of the Kalman filter). Additionally also
            alpha, V (smoothed state and smoothed state variance), and r and N
            (intermediary results) are outputted.
        Smoothed errors : several array-like items, optional
            Output of the Kalman smoother. The output consists of u, D (intermediary
            results), epsilon_hat, var_epsilon_cond,  eta_hat,  var_eta_cond
            observation and state errors and their variances.

        """
        
        #get state matrices         
        matrices, list_3d = self.get_matrices(self.matr)
        
        #apply Kalman filter
        at, Pt, a, P, v, F, K, newC, newD =self.kalman_filter_base(y, filter_init, self.matr)
        
        #initialise output arrays
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
        
        #flow if there are missing observations
        if np.isnan(np.sum(v)):
            #loop over the observations backwards
            for t in range(len(a)-3, -1,-1):
                #get the matrices at time t+1
                T, _, Z, _, _, _, _ = self.get_syst_matrices(list_3d, t+1, matrices.copy())
                args = v[t+1], F[t+1], r[t+1], T, K[t+1], Z, N[t+1], P[t+1], a[t+1]
                
                #if the observation is not missing, use the normal recursion
                if not np.isnan(v[t+1]):
                    L, r[t], N[t], alpha[t+1], V[t+1] = self.smoothing_iteration(*args)
            
                #use missing observation recursion if necessary
                else: 
                    L, r[t], N[t], alpha[t+1], V[t+1] = self.smoothing_iteration_missing(*args)
      
            #last recursion for alpha and V at time 0
            t = - 1
            T, _, Z, _, _, _, _ = self.get_syst_matrices(list_3d, t+1, matrices.copy())
            args = v[t+1], F[t+1], r[t+1], T, K[t+1], Z, N[t+1], P[t+1], a[t+1]

            #recursion if observation is not missing
            if not np.isnan(v[t+1]):
                _, _, _, alpha[t+1], V[t+1] = self.smoothing_iteration(*args)
     
            #recursion in case of missing observation
            else: 
                _, _, _, alpha[t+1], V[t+1] = self.smoothing_iteration_missing(*args)


        #flow if no missing observations                
        else: 
            #loop over the observations backwards
            for t in range(len(a)-3, -1,-1):
                T, _, Z, _, _, _, _ = self.get_syst_matrices(list_3d, t+1, matrices.copy())
                args = v[t+1], F[t+1], r[t+1], T, K[t+1], Z, N[t+1], P[t+1], a[t+1]
                
                L, r[t], N[t], alpha[t+1], V[t+1] = self.smoothing_iteration(*args)

            #recursion at time 0 for alpha and V
            t = - 1
            T, _, Z, _, _, _, _ = self.get_syst_matrices(list_3d, t+1, matrices.copy())
            args = v[t+1], F[t+1], r[t+1], T, K[t+1], Z, N[t+1], P[t+1], a[t+1]
            _, _, _, alpha[t+1], V[t+1] = self.smoothing_iteration(*args)

        #if the smoothed errors also need to be computed
        if return_smoothed_errors: 
            #compute smoothed erros
            u, D,  epsilon_hat, var_epsilon_cond,  eta_hat,  var_eta_cond = self.disturbance_smoothing_errors(v, F, K, r, N, matrices, list_3d)
            return at, Pt, a, P, v, F, K, newC, newD, alpha, V, r, N,  u, D,  epsilon_hat, var_epsilon_cond,  eta_hat,  var_eta_cond
     
        #only return smoother output
        else:
            return at, Pt, a, P, v, F, K, newC, newD, alpha, V, r, N
    
    
    def smoother(self,y, filter_init, return_smoothed_errors=True):
        """
        Wrapper around the smoother base function, to store results in a dict.

        Parameters
        ----------
        y : array-like
            Observation data.
        filter_init : tuple
            Initialisation of the Kalman filter.
        return_smoothed_errors : boolean, optional
            Indicates whether the smoothed errors also should be computed. 
            The default is True.

        Returns
        -------
        o : dict
            Kalman smoother output.
        e : dict
            Smoothed error output.
        """
        
        if return_smoothed_errors:
            o = {}
            e = {}
            o["at"], o["Pt"], o["a"], o["P"], o["v"], o["F"], o["K"], o["newC"], o["newD"], o["alpha"], o["V"], o["r"], o["N"], e['u'], e['D'],  e['epsilon_hat'], e['var_epsilon_cond'],  e['eta_hat'],  e['var_eta_cond']  =  self.smoother_base(y, filter_init)
            return {'output' : o, 'errors' : e}
        else:
            o = {}
            o["at"], o["Pt"], o["a"], o["P"], o["v"], o["F"], o["K"], o["newC"], o["newD"], o["alpha"], o["V"], o["r"], o["N"]  =  self.smoother_base(y, filter_init, return_smoothed_errors=False)
            return o
        
    
    def kalman_filter_CI(self, y, filter_init, conf=0.9):
        alpha_div2 = (1 - conf)/2
        n = norm.ppf(1 - alpha_div2) 
        
        o = {}
        o["at"], o["Pt"], o["a"], o["P"], o["v"], o["F"], o["K"], o["newC"], o["newD"] = self.kalman_filter_base(y, filter_init, self.matr)
        o['CI_filtered_' + str(conf) + "_lower"] = o["at"] - n*np.linalg.cholesky(o["Pt"])

        o['CI_filtered_' + str(conf)  + "_upper"] = o["at"] + n*np.linalg.cholesky(o["Pt"])
        
        return o
       
    
    def kalman_smoother_CI(self, y, filter_init, conf=0.9, return_smoothed_errors=True):
        alpha_div2 = (1 - conf)/2
        n = norm.ppf(1 - alpha_div2) 
        
        if return_smoothed_errors:
            o = {}
            e = {}
            o["at"], o["Pt"], o["a"], o["P"], o["v"], o["F"], o["K"], o["newC"], o["newD"], o["alpha"], o["V"], o["r"], o["N"], e['u'], e['D'],  e['epsilon_hat'], e['var_epsilon_cond'],  e['eta_hat'],  e['var_eta_cond']  =  self.smoother_base(y, filter_init)
        else:
            o = {}
            o["at"], o["Pt"], o["a"], o["P"], o["v"], o["F"], o["K"], o["newC"], o["newD"], o["alpha"], o["V"], o["r"], o["N"]  =  self.smoother_base(y, filter_init)

        o['CI_filtered_' + str(conf) + "_lower"] = o["at"] - n*np.linalg.cholesky(o["Pt"])
        o['CI_filtered_' + str(conf)  + "_upper"] = o["at"] + n*np.linalg.cholesky(o["Pt"])
        o['CI_smoothed_' + str(conf) + "_lower"] = o["alpha"] - n*np.linalg.cholesky(o["V"])
        o['CI_smoothed_' + str(conf)  + "_upper"] = o["alpha"] + n*np.linalg.cholesky(o["V"])
        
        try: 
            return {'output' : o, 'errors' : e}
        except NameError:
            return o
  
    
    def kalman_llik_base(self, param, y, matr, param_loc, filter_init, 
                    llik_fun = llik_gaussian, diffuse = 0):
        """
        Loglikelihood function for the Kalman filter system matrices. 
        The function allows for specification of the elements in the system matrices
        which are optimised, and which are remained fixed. A time-varying system
        matrix needs to have its parameters which are to be estimated by the 
        maximum likelihood estimator fixed for the whole period.

        Parameters
        ----------
        param : dict
            Parameter values tried.
        y : array-like
            Observation data.
        matr : dict
            System matrices used in the evaluation of the likelihood function.
        param_loc : dict
            Dictionnary with the locations and matrices of the parameters to 
            be optimised in the maximum likelihood function.
        filter_init : tuple
            initialisation of the filter.
        llik_fun : function, optional
            Function used to compute the log likelihood. 
            The default is llik_gaussian.
        diffuse : integer, optional
            Diffuse initialisation of the likelihood function. The default is 0.

        Returns
        -------
        llik_fun(v, F) : integer
            Evaluation of the log likelihood by the given function.

        """        
        
        #get the elements which are optimised in the ML function
        for key in param_loc.keys():
            matr[param_loc[key][0]][param_loc[key][1],param_loc[key][2]] = param[key]
        
        #apply Kalman Filter
        _, _, _, _, v, F, _, _, _  =  self.kalman_filter_base(y, filter_init, 
                                                              matr)
        
        #first element not used in diffuse likelihood
        v = v[diffuse:,:,:]
        F = F[diffuse:,:,:]

        return llik_fun(v, F)
        
    
    def kalman_llik(self, param, y, matr, param_loc, filter_init, 
                            llik_fun = llik_gaussian, diffuse=0):
        """
        Wrapper around the kalman_llik_base function where diffuse is set to 0.

        Parameters
        ----------
        param : dict
            Parameter values tried.
        y : array-like
            Observation data.
        matr : dict
            System matrices used in the evaluation of the likelihood function.
        param_loc : dict
            Dictionnary with the locations and matrices of the parameters to 
            be optimised in the maximum likelihood function.
        filter_init : tuple
            initialisation of the filter.
        llik_fun : function, optional
            Function used to compute the log likelihood. 
            The default is llik_gaussian.

        Returns
        -------
         self.kalman_llik_base( param, y, matr, param_loc, filter_init, 
                          llik_gaussian, diffuse = 1) : integer
            Evaluation of the log likelihood by the given function.

        """
        return self.kalman_llik_base(param, y, matr, param_loc, filter_init, 
                                llik_gaussian, diffuse = diffuse)

    
    def kalman_llik_diffuse(self, param, y, matr, param_loc, filter_init, 
                            llik_fun = llik_gaussian):
        """
        Wrapper around the kalman_llik_base function where diffuse is set to 1.

        Parameters
        ----------
        param : dict
            Parameter values tried.
        y : array-like
            Observation data.
        matr : dict
            System matrices used in the evaluation of the likelihood function.
        param_loc : dict
            Dictionnary with the locations and matrices of the parameters to 
            be optimised in the maximum likelihood function.
        filter_init : tuple
            initialisation of the filter.
        llik_fun : function, optional
            Function used to compute the log likelihood. 
            The default is llik_gaussian.

        Returns
        -------
         self.kalman_llik_base( param, y, matr, param_loc, filter_init, 
                          llik_gaussian, diffuse = 1) : integer
            Evaluation of the log likelihood by the given function.

        """
        return self.kalman_llik_base(param, y, matr, param_loc, filter_init, 
                                llik_gaussian, diffuse = 1)
    
    
    def fit(self, y, fit_method= ml_estimator_matrix, 
            matrix_order = ['T','R','Z','Q','H','c','d'], **fit_kwargs):
        """
        Fit function for estimating the system matrices based on the observations
        given. The function collects the parameters which are to be estimated 
        by looking for np.nan values in the system matrices. 

        Parameters
        ----------
        y : array-like
            Observation data.
        fit_method : function, optional
            Function for the estimation of the parameters. 
            The default is ml_estimator_matrix.
        matrix_order : list, optional
            order of the system matrices. The default is ['T','R','Z','Q','H','c','d'].
        **fit_kwargs : dict
            additional arguments necessary for running the fit function.

        Returns
        -------
        self : state_spacer object
            object where the parameters of the state matrices are estimated.

        """
        
        #make a dict which contains all parameter locations in the system 
        #matrices which need to be estimated
        param_loc = {}
        
        #go through teh system matrices
        i=0
        for key in  (matrix_order):
            #get the elements which are np.nan
            nan_location = np.argwhere(np.isnan(self.matr[key]))[:,[0,1]]
            if len(nan_location):
                nan_location = np.unique(np.argwhere(np.isnan(self.matr[key]))[:,[0,1]],axis=0)

            #add the matrix, as well as the location in the matrix to the dict
            #with locations
            for loc in nan_location:
                param_loc[i] = key, loc[0], loc[1]
                i += 1

        #apply the fit method to the system matrices
        res = fit_method(y, self.matr, param_loc, **fit_kwargs)
        
        #get the results of the optimisation
        param = res.x

        #get the elements which are optimised in the fit function
        for key in param_loc.keys():
            self.matr[param_loc[key][0]][param_loc[key][1],param_loc[key][2]] = param[key]
        
        #set boolean showing if the model is fitted to true
        self.fitted = True
        
        #store information about the fit procedure in the object
        self.fit_parameters["fit_method"] = fit_method
        self.fit_parameters["matrix_order"] = matrix_order
        for kwarg in fit_kwargs.keys():
            self.fit_parameters[str(kwarg)] = fit_kwargs[kwarg]
        self.fit_parameters["param_loc"] = param_loc
        self.fit_results = res
        
        #return object
        return self


    def disturbance_smoothing_errors_iteration(self, H, Q, R, v, F, K, r, N):
        """
        Computation of the smoothing errors

        Parameters
        ----------
        H : np.matrix
            System matrix Ht.
        Q : np.matrix
            System matrix Qt.
        R : np.matrix
            System matrix Rt.
        v : array-like
            Prediction error.
        F : array-like
            Prediction error variance.
        K : array-like
            K (Kalman filter output).
        r : array-like
            r (Kalman smoothing output).
        N : array-like
            N (Kalman smoothing output).

        Returns
        -------
        u : array-like
            u (intermediary result) of time t.
        D : array-like
            D (intermediary result) of time t.
        epsilon_hat : array-like
            Estimated observation error of time t.
        var_epsilon_cond : array-like
            Estimated observation error variance of time t.
        eta_hat : array-like
            Estimated state error of time t.
        var_eta_cond : array-like
            Estimated state error variance of time t.

        """
        #convert arrays to matrices
        v = np.matrix(v)
        F = np.matrix(F)
        K = np.matrix(K)
        r = np.matrix(r)
        N = np.matrix(N)
        
        # calculate u = v_t*F_t^-1 - K_t*r_t
        u = v * np.linalg.inv(F) - r * K
    
        # calculate D_t = F_t^-1 + K_t* N_t*K_t
        D = np.linalg.inv(F) + np.transpose(K) * N * K

        # estimated epsilon_t= H * u_t
        epsilon_hat =  u * np.transpose(H)
    
        # estimated conditional variance_t epsilon = H - H*D_t *H
        var_epsilon_cond = H - H* D * H
    
        # estimated eta_t= Q*R' * r_t
        eta_hat = r * R * np.transpose(Q)
    
        # estimated conditional variance_t eta = Q - Q*R'* N_t *R*Q
        var_eta_cond = Q - Q * np.transpose(R) * N * R * Q
    
        return  u, D,  epsilon_hat, var_epsilon_cond,  eta_hat,  var_eta_cond
    

        

    def disturbance_smoothing_errors(self,  v, F, K, r, N, matrices, list_3d):
        """
        Function regulating the flow of computing the smoothingerrors

        Parameters
        ----------
        v : array-like
            Prediction error.
        F : array-like
            Prediction error variance.
        K : array-like
            K (Kalman filter output).
        r : array-like
            r (Kalman smoothing output).
        N : array-like
            N (Kalman smoothing output).
        matrices : dict
            System matrices.
        list_3d : list
            List of 3D matrices.

        Returns
        -------
        u : array-like
            u (intermediary result).
        D : array-like
            D (intermediary result).
        epsilon_hat : array-like
            Estimated observation error.
        var_epsilon_cond : array-like
            Estimated observation error variance.
        eta_hat : array-like
            Estimated state error.
        var_eta_cond : array-like
            Estimated state error variance.

        """
        #get the first system matrices for setting the array dimensions
        _, _, _, Q, H, _, _ = self.get_syst_matrices(list_3d, 0, matrices.copy())

        #get the length of the series
        time = len(v)
        
        #initialisation of the arrays
        u   = np.zeros((time, (v).shape[1], (np.linalg.inv(F)).shape[1]))
        D = np.zeros((time, np.linalg.inv(F).shape[1], np.linalg.inv(F).shape[1]))
        epsilon_hat = np.zeros((time,  v.shape[1], H.shape[1]))
        var_epsilon_cond = np.zeros((time, H.shape[0], H.shape[1]))
        eta_hat = np.zeros((time,  r.shape[1], Q.shape[1]))
        var_eta_cond = np.zeros((time,  (Q).shape[0], (Q).shape[1]))
        
        #for each time, compute the errors
        for t in range(len(v)):
            _, R, _, Q, H, _, _ = self.get_syst_matrices(list_3d, t, matrices.copy())
            u[t], D[t],  epsilon_hat[t], var_epsilon_cond[t],  eta_hat[t],  var_eta_cond[t] = self.disturbance_smoothing_errors_iteration(H, Q, R, v[t], F[t], K[t], r[t], N[t])
    
        return  u, D,  epsilon_hat, var_epsilon_cond,  eta_hat,  var_eta_cond


    def simulation_smoother_one(self, y, filter_init, eta, epsilon, dist_fun_alpha1):
        """
        Implementation of the simulation smoother. Compute a single path of 
        alpha_tilde.

        Parameters
        ----------
        y : array-like
            Observation data.
        filter_init : tuple
            filter initalisation.
        eta : array-like
            Series of simulated state errors.
        epsilon : array-like
            Series of simulated observation errors.
        dist_fun_alpha1 : function
            distribution of the first element of alpha.

        Returns
        -------
        alpha_tilde : simulated path of alpha.

        """
        #get matrices
        matrices, list_3d = self.get_matrices(self.matr)
        
        #determine the number of states
        states = self.matr['T'].shape[1]
        
        #initialise arrays
        alphaplus = np.zeros((len(y), states))
        yplus = np.zeros(y.shape)
    
        #unpack filter initialisation
        a1, P1 = filter_init
        
        #get first system matrices
        t=0
        T, R, Z, Q, H, c, d = self.get_syst_matrices(list_3d, t, matrices.copy())
        
        #compute the alpha+1 and y+1
      #  alphaplus[t,:] = (d + np.matrix(np.random.multivariate_normal(a1,np.linalg.cholesky(np.matrix(P1)))).T).reshape(-1)
        alphaplus[t,:] = (d + np.matrix(dist_fun_alpha1(a1,np.linalg.cholesky(np.matrix(P1)))).T).reshape(-1)
     #   alphaplus[t] = d + np.matrix(np.random.normal(a1,np.linalg.cholesky(np.matrix(P1)),size=(alphaplus[t].shape))).reshape(-1)
        yplus[t] = c + alphaplus[t]*np.transpose(Z)  + np.linalg.cholesky(H)*epsilon[t]
        
        #create alpha+ and y+ iteratively
        for t in range(len(alphaplus)-1):
            T, R, Z, Q, H, c, d = self.get_syst_matrices(list_3d, t, matrices.copy())
            alphaplus[t+1] = np.transpose(d) +  alphaplus[t]*np.transpose(T) + np.transpose(R*np.linalg.cholesky(Q)*eta[t].reshape(-1,1))
           # alphaplus[t+1] = np.transpose(d) +  alphaplus[t]*np.transpose(T) + np.transpose(eta[t].reshape(-1,1))*np.transpose(np.linalg.cholesky(Q))*np.transpose(R)
            yplus[t+1] = np.transpose(c) + alphaplus[t+1]*np.transpose(Z) + np.transpose(epsilon[t+1])*np.transpose(np.linalg.cholesky(H))
            
        #compute y_tilde
        y_tilde = y - yplus

        #run the KFS on y_tilde
        at, Pt, a, P, v, F, K, newC, newD, alpha, V, r, N = self.smoother_base(y_tilde, filter_init,return_smoothed_errors=False)

        #compute alpha tilde        
        alpha_tilde = alphaplus + alpha.reshape(-1,states)

        return alpha_tilde

    
    def simulation_smoother(self, y, filter_init, n, dist_fun_alpha1=None, 
                            **kwargs):
        """
        Computes n paths of alpha_tilde via simulation smoothing.        

        Parameters
        ----------
        y : array-like
            Observation data.
        filter_init : tuple
            filter initalisation.
        n : integer
            number of paths to be simulated.
        dist_fun_alpha1 : function, optional
            distribution of alpha1. The default is None.
        **kwargs : dict
            DESCRIPTION.

        Returns
        -------
        alpha_tilde_array : array-like
            array of simulated paths of alpha_tilde.

        """
        #determine the number of states
        states = self.matr['T'].shape[1]
        
        #in case no distribution is given, alpha1 is assumed to be normally 
        #distributed (potentially multivariate).
        if dist_fun_alpha1 is None:
            if self.matr['R'].shape[1] > 1:
                dist_fun_alpha1 = np.random.multivariate_normal
            else:
                dist_fun_alpha1 = np.random.normal
        
        #change the form of the error term array to have n simulations
        eps_shape = list(y.shape)
        eps_shape.append(n)
        eps_shape = tuple(eps_shape)
        
        #simulate error terms
        epsilon = np.random.normal(0,1, size=eps_shape)
        eta = np.random.normal(0,1, size=(len(y),self.matr['R'].shape[1],n))
                
        #initialise the alpha tilde array
        alpha_tilde_array = np.zeros((len(y), states,n))

        #compute the smoothed paths one by one
        for i in range(n):
            alpha_tilde_array[:,:,i] = self.simulation_smoother_one(y, filter_init, eta[:,:,i], epsilon[:,:,i], dist_fun_alpha1, **kwargs)
        
        return alpha_tilde_array
    
    
    def simulation_smoother_2(self, y, filter_init, n, dist_fun_alpha1=None, 
                            **kwargs):
            """
            Computes n paths of alpha_tilde via simulation smoothing.        
    
            Parameters
            ----------
            y : array-like
                Observation data.
            filter_init : tuple
                filter initalisation.
            n : integer
                number of paths to be simulated.
            dist_fun_alpha1 : function, optional
                distribution of alpha1. The default is None.
            **kwargs : dict
                DESCRIPTION.
    
            Returns
            -------
            alpha_tilde_array : array-like
                array of simulated paths of alpha_tilde.
    
            """
            #determine the number of states
            states = self.matr['T'].shape[1]
            
            #in case no distribution is given, alpha1 is assumed to be normally 
            #distributed (potentially multivariate).
            if dist_fun_alpha1 is None:
                if self.matr['R'].shape[1] > 1:
                    dist_fun_alpha1 = np.random.multivariate_normal
                else:
                    dist_fun_alpha1 = np.random.normal
            
            #change the form of the error term array to have n simulations
            eps_shape = list(y.shape)
            eps_shape.append(n)
            eps_shape = tuple(eps_shape)
            
            #simulate error terms
            epsilon = np.random.normal(0,1, size=eps_shape)
            eta = np.random.normal(0,1, size=(len(y),self.matr['R'].shape[1],n))
                    
            #initialise the alpha tilde array
            alpha_tilde_array = np.zeros((len(y), states,n))

            #initialise arrays
            alphaplus = np.zeros((len(y), states,n))
            yplus = np.zeros((y.shape[0], y.shape[1],n))
              
            alphaplus, yplus = self.simulation_smoother_three(dist_fun_alpha1,filter_init,n, alphaplus, yplus, epsilon, eta, **kwargs)
            at, Pt, a, P, v, F, K, newC, newD, alpha_orig, V, r, N  = self.smoother_base(y, filter_init,return_smoothed_errors=False)
            #compute the smoothed paths one by one
            for i in range(n):
                y_tilde = y - yplus[:,:,i]

                #run the KFS on y_tilde
                at, Pt, a, P, v, F, K, newC, newD, alpha, V, r, N = self.smoother_base(y_tilde, filter_init,return_smoothed_errors=False)
                #compute alpha tilde      
                alpha_tilde_array[:,:,i] = alphaplus[:,:,i] + alpha.reshape(-1,states)


                #alpha_tilde_array[:,:,i] = self.simulation_smoother_three(y, filter_init, alphaplus[:,:,i], epsilon[:,:,i], dist_fun_alpha1, **kwargs)
            
            return alpha_tilde_array
 
    
    def simulation_smoother_three(self,dist_fun_alpha1,filter_init,n, alphaplus, yplus, epsilon, eta):

        matrices, list_3d = self.get_matrices(self.matr)
        a1, P1 = filter_init
        a1 = np.zeros((np.array(a1).shape))
        t=0
        T, R, Z, Q, H, c, d = self.get_syst_matrices(list_3d, t, matrices.copy())
        alphaplus[t]  = dist_fun_alpha1(a1,np.linalg.cholesky(np.matrix(P1)),size=(n)).T
        yplus[t] =  Z*alphaplus[t]  + epsilon[t]
      #  alphaplus[t]  = dist_fun_alpha1(a1,np.linalg.cholesky(np.matrix(P1)),size=(n)).T
      #  yplus[t] = c + Z*alphaplus[t]  + epsilon[t]
        
        for t in range(len(alphaplus)-1):
            T, R, Z, Q, H, c, d = self.get_syst_matrices(list_3d, t, matrices.copy())
            eta[t] = np.linalg.cholesky(Q)*np.matrix(eta[t])       
            epsilon[t] = np.linalg.cholesky(H)*np.matrix(epsilon[t]) 
            
            alphaplus[t+1] =  T*alphaplus[t] + R*eta[t]
            yplus[t+1] =  Z*alphaplus[t+1] + epsilon[t+1]
            
            #alphaplus[t+1] = d + T*alphaplus[t] + R*eta[t]
            # alphaplus[t+1] = np.transpose(d) +  alphaplus[t]*np.transpose(T) + np.transpose(eta[t].reshape(-1,1))*np.transpose(np.linalg.cholesky(Q))*np.transpose(R)
            #yplus[t+1] = c + Z*alphaplus[t+1] + epsilon[t+1]

        return alphaplus, yplus

    
    # A method for saving object data to JSON file
    def save_json(self, filepath):
        """
        Method for saving a state_spacer object on a file path. Works not 
        perfectly, as some of the optimization parts cannot easily be stored

        Parameters
        ----------
        filepath : string
            filepath where object is saved.

        Returns
        -------
        None.

        """
        
        #stringify 
        self.fit_results['message'] = str(self.fit_results['message'])

        #make dict with the necessary elements
        dict_ = {}
        dict_['fitted'] = self.fitted
        dict_['matr'] = {}
        dict_['fit_results'] = {}
        dict_['fit_parameters'] = {}
        
        #save all the save matrices
        for matrix in self.matr.keys():
            dict_['matr'][matrix] = self.matr[matrix].tolist()
        
        #save the fit results (bit ugly because of their structure)
        for key in self.fit_results.keys():
            if key == "hess_inv":
                dict_['fit_results'][key] = {}
                for hess_key in  self.fit_results[key].__dict__.keys():
                    if type(self.fit_results[key].__dict__[hess_key]) is not np.dtype:
                        dict_['fit_results'][key][hess_key] = self.fit_results[key].__dict__[hess_key]
            else: 
                dict_['fit_results'][key] = self.fit_results[key]
                
        #add the parameter keys
        for key in self.fit_parameters.keys():
            if callable(self.fit_parameters[key]):
                dict_['fit_parameters'][key] = self.fit_parameters[key].__name__
            else:
                dict_['fit_parameters'][key] = self.fit_parameters[key]
                
        
        # Creat json and save to file
        json_txt = json.dumps(dict_, cls=NpEncoder, indent=4)
        
        with open(filepath, 'w') as file:
            file.write(json_txt)
        print('Object stored under ' + filepath)
    
    # A method for loading data from JSON file
    def load_json(self, filepath):
        """
        method for loading state space model by a json file.

        Parameters
        ----------
        filepath : string
            file path of the object.

        Returns
        -------
        None.

        """
        #read in json data
        with open(filepath, 'r') as file:
            dict_ = json.load(file)
            
        #get fitted info 
        self.fitted = dict_['fitted'] 
        
        #get matrix info
        for matrix in dict_['matr'].keys():
            self.matr[matrix] = np.asarray(dict_["matr"][matrix]) if dict_["matr"][matrix] != 'None' else None

        #get fit results info
        for key in dict_['fit_results'].keys():
            self.fit_results[key] = dict_['fit_results'][key]  if dict_['fit_results'][key]  != 'None' else None

        #get fit parameter info
        for key in dict_['fit_parameters'].keys():
            self.fit_parameters[key] = dict_['fit_parameters'][key] if dict_['fit_parameters'][key] != 'None' else None


    def save_matrices_json(self, filepath):
        """
        Method which only saves the matrices in a json file.

        Parameters
        ----------
        filepath : string
            file path where object is stored.

        Returns
        -------
        None.

        """
        
        dict_ = {}
        
        #save matrices
        for matrix in self.matr.keys():
            dict_['matr'][matrix] = self.matr[matrix].tolist()
            
        
        # Creat json and save to file
        json_txt = json.dumps(dict_, cls=NpEncoder, indent=4)
        
        #write info in file
        with open(filepath, 'w') as file:
            file.write(json_txt)
            
    
    def load_matrices(self, filepath, previously_fitted=False):
        """
        Method which only reads in the system matrices and overwrites all other
        info.

        Parameters
        ----------
        filepath : string
            file path where object is stored.
        previously_fitted : boolean, optional
            Whether or not the matrices were fitted before. The default is False.

        Returns
        -------
        None.

        """
   
        #open filepath
        with open(filepath, 'r') as file:
            dict_ = json.load(file)
            
        #read in the matrices
        for matrix in dict_['matr'].keys():
            self.matr[matrix] = np.asarray(dict_["matr"][matrix]) if dict_["matr"][matrix] != 'None' else None

        self.fit_parameters = {}
        self.fit_results = {}
        self.fitted = previously_fitted



#helper class for json encoding
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

