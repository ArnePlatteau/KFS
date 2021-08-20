# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 20:12:18 2021

@author: arnep
"""

from kalman import state_spacer as StateSpacer
import numpy as np

class eStateSpacer(StateSpacer):
    """Class which implements the extended Kalman Filter. This technique 
    uses a Taylor transformation of the first order for linearising non-linear
    functions and then uses the Kalman filter on the linearised function.
    """
    def __init__(self, ZFun, ZDotFun, TFun, TDotFun, ZArgs = {}, ZDotArgs = {},
                 TArgs = {}, TDotArgs = {}, *matrices
                 ):
        """
        Implementation of the following model:
            yt = c + Z(alphat) + epst , epst ~ NID(0,H)
            alphat+1 = d + T(alphat) + Rt*etat , etat ~NID(0,Q) 
        
        With Z(), and T() differentiable (non-)linear functions
        
        define time-varying structural matrices in dimension (row, column, time)


        Parameters
        ----------
        ZFun : function
            Definition of the Z function.
        ZDotFun : function
            Definition of the derivative of the Z function.
        TFun : function
            Definition of the T function.
        TDotFun : function
            Definition of the derivative of the T function.
        ZArgs : dict, optional
            Additional arguments for the Z function. 
            The default is {}.
        ZDotArgs : dict, optional
            Additional arguments for the derivative of the Z function. 
            The default is {}.
        TArgs : dict, optional
            Additional arguments for the T function. 
            The default is {}.
        TDotArgs : dict, optional
            Additional arguments for the derivative of the T function. 
            The default is {}.
        *matrices : dict
                    System matrices of the state space model.

        Returns
        -------
        None.

        """
        self.ZFun = ZFun
        self.ZDotFun = ZDotFun
        self.TFun = TFun
        self.TDotFun = TDotFun

        self.ZArgs = ZArgs
        self.ZDotArgs = ZDotArgs
        self.TArgs = TArgs
        self.TDotArgs = TDotArgs
        super().__init__(*matrices)


    def get_Z(self, x, *args):
        """
        Evaluate the Z function at x.

        Parameters
        ----------
        x : input of the Z function.
        *args : additional arguments for the Z function.

        Returns
        -------
        int
            evaluation of the Z function.

        """
        return np.matrix(self.ZFun(x, *args, **self.ZArgs))


    def get_Z_dot(self, x, *args):
        """
        Evaluate the derivative of the Z function at x.

        Parameters
        ----------
        x : input of the derivative of the Z function.
        *args : additional arguments for the derivative of the Z function.

        Returns
        -------
        int
            evaluation of the derivative of the Z function.

        """
        return np.matrix(self.ZDotFun(x, *args, **self.ZDotArgs))


    def get_T(self, x, *args):
        """
        Evaluate the T function at x.

        Parameters
        ----------
        x : input of the T function.
        *args : additional arguments for the T function.

        Returns
        -------
        int
            evaluation of the T function.


        """
        return np.matrix(self.TFun(x, *args, **self.TArgs))


    def get_T_dot(self, x, *args):
        """
        Evaluate the derivative of the T function at x.

        Parameters
        ----------
        x : input of the derivative of the T function.
        *args : additional arguments for the derivative of the T function.

        Returns
        -------
        int
            evaluation of the derivative of the T function.

        """

        return np.matrix(self.TDotFun(x, *args, **self.TDotArgs))
    

    def kalman_filter_iteration(self, yt, a, P, Z, T, c, d, H, Q, R,
                                v, F, att, Ptt):
        """
        Single iteration of the Kalman filter.         
        v_t = y_t - Z(a) - c_t 
        F_t = Zdot_t*P_t* Zdot_t' +  H_t
        K_t = T_t*P_t*Zdot_t'*F_t-1
        a_{t+1} = T(at) + K_t*v_t + d
        P_{t+1} = Tdot_t*P_t*Tdot_t' + R_t*Q_t*R_t' - K_t*F_t*K_t' 


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


        Z_new = np.matrix(self.get_Z_dot(a, Z))

        c = np.matrix(c + self.get_Z(a, Z) - a*self.get_Z_dot(a, Z).transpose())

        #v and a are transposed
        v = yt -a*Z_new.transpose() - c.transpose() 
        #F, P and K are not transposed
        F = Z_new*P*Z_new.transpose() + H
        M = P*Z_new.transpose()*np.linalg.inv(F)
        K = T*M
        
        att = a + v*M.transpose()
        Ptt = P - M.transpose()*P*M
        
        T_new = np.matrix(self.get_T_dot(att, T))        
        d = np.matrix(d + self.get_T(att, T).transpose() - self.get_T_dot(att, T)*att.transpose())
        
        at1 = a*T_new.transpose() + v*K.transpose() + d.transpose()
        Pt1 = T_new*P*T_new.transpose() + R*Q*R.transpose() - K*F*K.transpose()
        
        return v, F, K, att, Ptt, at1, Pt1, c, d 


    def kalman_filter_iteration_missing(self, yt, a, P, Z, T, c, d, H, Q, R,
                                v, F, att, Ptt, tol = 1e7 ):
        """
        Kalman iteration function in case the observation is missing.
        
        v_t = undefined
        F_t = infinity
        K_t = 0
        a_{t+1} = T(a_t) + d
        P_{t+1} = Tdot*P_t*Tdot_t' + R_t*Q_t*R_t'

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
        
        c = np.matrix(c + self.get_Z(a, Z) - a*self.get_Z_dot(a, Z).transpose())

        #v and a are transposed
        v = yt
        #F, P and K are not transposed
        F = np.matrix(np.ones(H.shape)*tol)
        M = np.matrix(np.zeros((P*Z.transpose()*np.linalg.inv(F)).shape)) #set zeros
        K = T*M
        
        att = a 
        Ptt = P
        
        T_new = np.matrix(self.get_T_dot(att, T))        
        d = np.matrix(d + self.get_T(att, T).transpose() - self.get_T_dot(att, T)*att.transpose())
        
        at1 = a*T_new.transpose() + d.transpose()
        Pt1 = T_new*P*T_new.transpose() + R*Q*R.transpose()


        return v, F, K, att, Ptt, at1, Pt1, c, d

    
        def simulation_smoother_one(self, y, filter_init, eta, epsilon, 
                                    dist_fun_alpha1, alpha_fun, y_fun):
            """work in progress"""
            matrices, list_3d = self.get_matrices(self.matr)
    
            a1, P1 = filter_init
            states = self.matr['T'].shape[1]
            alphaplus = np.zeros((len(y), states))
            yplus = np.zeros(y.shape)
            
            t=0
            T, R, Z, Q, H, c, d = self.get_syst_matrices(list_3d, t, matrices.copy())
          #  alphaplus[t,:] = (d + np.matrix(np.random.multivariate_normal(a1,np.linalg.cholesky(np.matrix(P1)))).T).reshape(-1)
            alphaplus[t,:] = (d + np.matrix(dist_fun_alpha1(a1,np.linalg.cholesky(np.matrix(P1)))).T).reshape(-1)
    
         #   alphaplus[t] = d + np.matrix(np.random.normal(a1,np.linalg.cholesky(np.matrix(P1)),size=(alphaplus[t].shape))).reshape(-1)
            yplus[t] = c + alphaplus[t]*np.transpose(Z)  + np.linalg.cholesky(H)*epsilon[t]
            for t in range(len(alphaplus)-1):
                T, R, Z, Q, H, c, d = self.get_syst_matrices(list_3d, t, matrices.copy())
                alphaplus[t+1] = np.transpose(d) +  alphaplus[t]*np.transpose(T) + np.transpose(R*np.linalg.cholesky(Q)*eta[t].reshape(-1,1))
    
               # alphaplus[t+1] = np.transpose(d) +  alphaplus[t]*np.transpose(T) + np.transpose(eta[t].reshape(-1,1))*np.transpose(np.linalg.cholesky(Q))*np.transpose(R)
                yplus[t+1] = np.transpose(c) + alphaplus[t+1]*np.transpose(Z) + np.transpose(epsilon[t+1])*np.transpose(np.linalg.cholesky(H))
                
            y_tilde = y - yplus
    
           # at, Pt, a, P, v, F, K, newC, newD, alpha, V, r, N = self.smoother_base(y_tilde, filter_init,return_smoothed_errors=False)
            o = {}
            o["at"], o["Pt"], o["a"], o["P"], o["v"], o["F"], o["K"], o["newC"], o["newD"], o["alpha"], o["V"], o["r"], o["N"] =  self.smoother_base(y_tilde, filter_init,return_smoothed_errors=False)
            alpha = o["alpha"]
            alpha_tilde = alphaplus + alpha.reshape(-1,states)
    
            return alpha_tilde
    

        def simulation_smoother(self, y, filter_init, n, dist_fun_alpha1=None, **kwargs):
            """Work in progress """
            print('Not implemented yet.')
            
