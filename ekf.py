# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 20:12:18 2021

@author: arnep
"""

from kalman import state_spacer as StateSpacer
import numpy as np

class eStateSpacer(StateSpacer):
    """Class which implements the extended Kalman Filter. This technique 
    uses a Taylor transformation
    """
    def __init__(self, ZFun, ZDotFun, TFun, TDotFun, ZArgs = {}, ZDotArgs = {},
                 TArgs = {}, TDotArgs = {}, *matrices
                 ):
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
        return np.matrix(self.ZFun(x, *args, **self.ZArgs))


    def get_Z_dot(self, x, *args):
        return np.matrix(self.ZDotFun(x, *args, **self.ZDotArgs))


    def get_T(self, x, *args):
        return np.matrix(self.TFun(x, *args, **self.TArgs))


    def get_T_dot(self, x, *args):
        return np.matrix(self.TDotFun(x, *args, **self.TDotArgs))
    

    def kalman_filter_iteration(self, yt, a, P, Z, T, c, d, H, Q, R,
                                v, F, att, Ptt):
        """
        v_t = y_t - Z_t*a_t - c_t
        F_t = Z_t*P_t* Z_t' +  H_t
        K_t = T_t*P_t*Z_t'*F_t-1
        a_{t+1} = T_t* a_t + K_t*v_t + d
        P_{t+1} = T*P_t*T_t' + R_t*Q_t*R_t' - K_t*F_t*K_t' 
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
        v_t = y_t - Z_t*a_t - c_t
        F_t = Z_t*P_t* Z_t' +  H_t
        K_t = T_t*P_t*Z_t'*F_t-1
        a_{t+1} = T_t* a_t + K_t*v_t + d
        P_{t+1} = T*P_t*T_t' + R_t*Q_t*R_t' - K_t*F_t*K_t' 
        """
        
        c = np.matrix(c + self.get_Z(a, Z) - a*self.get_Z_dot(a, Z).transpose())

        #v and a are transposed
        v = yt
        #F, P and K are not transposed
        F = np.matrix(np.ones(H.shape)*tol)
        M = np.matrix(np.zeros((P*Z.transpose()*np.linalg.inv(F)).shape))
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
            print('Not implemented yet.')
            
