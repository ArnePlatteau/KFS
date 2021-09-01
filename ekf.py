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


        """
        c = np.matrix(c + self.get_Z(a, Z) - a*self.get_Z_dot(a, Z).transpose())
        
        #v and a are transposed
        v = yt -a*Z_new.transpose() - c.transpose() 
        #F, P and K are not transposed
        F = Z_new*P*Z_new.transpose() + H
        M = P*Z_new.transpose()*np.linalg.inv(F)
        K = T*M
        
        att = a + v*M.transpose()
        Ptt = P - M.transpose()*F*M
        
        T_new = np.matrix(self.get_T_dot(att, T))        
        d = np.matrix(d + self.get_T(att, T).transpose() - self.get_T_dot(att, T)*att.transpose())
        
        at1 = a*T_new.transpose() + v*K.transpose() + d.transpose()
        Pt1 = T_new*P*T_new.transpose() + R*Q*R.transpose() - K*F*K.transpose()
        """
      
        T_new = np.matrix(self.get_T_dot(att, T))        
        Z_new = np.matrix(self.get_Z_dot(a, Z))
        
        v = yt - self.get_Z(a, Z).transpose() - c.transpose() 
        #F, P and K are not transposed
        F = Z_new*P*Z_new.transpose() + H
        M = P*Z_new.transpose()*np.linalg.inv(F)
        K = T_new*M
        
        att = a + v*M.transpose()
        Ptt = P - M.transpose()*F*M
        
        #     d = np.matrix(d + self.get_T(att, T).transpose() - self.get_T_dot(att, T)*att.transpose())
        
        at1 = self.get_T(att, T) + d
        Pt1 = T_new*Ptt*T_new.transpose() + R*Q*R.transpose() 
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
        """


        T_new = np.matrix(self.get_T_dot(att, T))        
        Z_new = np.matrix(self.get_Z_dot(a, Z))
        
        v = yt*np.nan
        #F, P and K are not transposed
        F = np.matrix(np.ones(H.shape)*tol)
        M = np.matrix(np.zeros((P*Z_new.transpose()*np.linalg.inv(F)).shape)) #set zeros
        K = T_new*M
        
        att = a 
        Ptt = P
        
        
        at1 = self.get_T(att, T) + d
        Pt1 = T_new*Ptt*T_new.transpose() + R*Q*R.transpose() 

        return v, F, K, att, Ptt, at1, Pt1, c, d


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
        Z_new = np.matrix(self.get_Z_dot(a, Z))

        M = P*Z_new.transpose()*np.linalg.inv(F)
        att = a + v*M.transpose()
        
        T_new = np.matrix(self.get_T_dot(att, T))        
        

        L_new = T_new - K*Z_new
        r= v*np.linalg.inv(F).transpose()*Z_new + (r*L_new)
        N = Z_new.transpose()*np.linalg.inv(F)*Z_new + L_new.transpose()*N*L_new
        alpha = a + np.dot(r,P.transpose())
        V = P - P*N*P
        return L_new, r, N, alpha, V


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
        Z_new = np.matrix(self.get_Z_dot(a, Z))
        M = P*Z_new.transpose()*np.linalg.inv(F)        
        att = a + v*M.transpose()
        T_new = np.matrix(self.get_T_dot(att, T))        
        
        L_new = T_new
        r= r*L_new
        N = L_new.transpose()*N*L_new
        alpha = a + np.dot(r,P.transpose())
        V = P - P*N*P
        
        return L_new, r, N, alpha, V

    
    def simulation_smoother_three(self,dist_fun_alpha1,filter_init,n, alphaplus, yplus, epsilon, eta):

        matrices, list_3d = self.get_matrices(self.matr)
        a1, P1 = filter_init
        
        a1 = np.zeros((np.array(a1).shape))

        t=0
        T, R, Z, Q, H, c, d = self.get_syst_matrices(list_3d, t, matrices.copy())
        alphaplus[t] = dist_fun_alpha1(a1,P1,size=(n)).T  
        yplus[t] = self.ZFun(alphaplus[t+1].transpose(), Z).transpose() + epsilon[t+1]
        
        for t in range(len(alphaplus)-1):
            T, R, Z, Q, H, c, d = self.get_syst_matrices(list_3d, t, matrices.copy())
            eta[t] = np.linalg.cholesky(Q)*np.matrix(eta[t])       
            epsilon[t] = np.linalg.cholesky(H)*np.matrix(epsilon[t]) 
            
            alphaplus[t+1] =  self.get_T(alphaplus[t].transpose(),T).transpose() + R*eta[t]
            yplus[t+1] =  self.get_Z(alphaplus[t+1].transpose(), Z).transpose() + epsilon[t+1]

        return alphaplus, yplus

    

        def simulation_smoother(self, y, filter_init, n, dist_fun_alpha1=None, **kwargs):
            """Work in progress """
            print('Not implemented yet.')
            
