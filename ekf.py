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

