#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:02:02 2019

@author: dm
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 14:17:15 2019

@author: dm
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D  
#from matplotlib import ticker, cm

"""
Random correlation matrix generator
"""
def generate_random_correlation_matrix(factor_count, dimension):
    d = dimension
    k = factor_count
    W = np.random.normal(0, 1, (d,k))
    A = np.dot(W, W.T)
    D = np.zeros((d,d))
    for i in range(0, d) : 
        D[i, i] = np.random.uniform(0, 1)
    S = A + D    
    S_sq = np.zeros((d,d))
    for i in range(0, d) :
        S_sq[i,i] = 1 / np.sqrt(S[i,i])
    
    C_m = np.dot(S_sq, S)
    C_m = np.dot(C_m, S_sq)
    return C_m


"""
Class of multidimensional Ornstein-Uhlenbeck process
"""
class OU_process :  
    """
    The process of the form : 
        dX_t = -\kappa (X_t - \theta) dt + \sigma dB_t 
    
    n - dimension
    kappa - a diagonal matrix of reversion speeds  
    sigma - a diagonal matrix of noise magnitudes
    sigma_inv - the inverse to the \sigma matrix
    theta - a vector of long-term means 
    corr_matrix - Correlation matrix for a n-dimensional Wiener process B_t     
    """    
    n = None
    kappa = None
    sigma = None
    sigma_inv = None
    theta = None
    corr_matrix = None

    def __init__(self, kappa_vec, sigma_vec, corr_matrix, theta) :
        self.n = len(kappa_vec)
        self.kappa = np.diag(kappa_vec)
        self.sigma = np.diag(sigma_vec)
        self.sigma_inv = np.linalg.inv(self.sigma)
        self.corr_matrix = corr_matrix.copy()
        self.theta = np.reshape(theta, (self.n, 1))

    """
    simulate trajectories of OU process 
    x0 - initial point
    T - terminal time 
    dt - time step
    Output - a matrix of the size T/dt \times self.n 
    """    
    def simulate(self, x0, T, dt) : 
        t = np.arange(0, T, dt)
        sample_sz = len(t)
        n = self.n
        W = np.zeros((sample_sz, n))
        bm_increments = np.random.multivariate_normal(np.zeros(n), self.corr_matrix, sample_sz)
        
        for i in range(0, sample_sz -1) :
            for j in range(0, n) :
                j_kappa = self.kappa[j, j]
                if j_kappa != 0 : 
                    W[i + 1, j] = W[i, j] + np.sqrt(np.exp(2 * j_kappa * t[i + 1]) - np.exp(2 * j_kappa * t[i])) * bm_increments[i,j]
                else : 
                    W[i + 1, j] = W[i, j] + np.sqrt(dt) * bm_increments[i, j]
        X = np.zeros((sample_sz, n))
        ex = np.repeat(np.reshape(t, (sample_sz, 1)), n, axis = 1);
        ex = np.exp(-np.dot(ex, self.kappa))
        
        for i in range(0, n) :
            if self.kappa[i, i] != 0 :
                X[:, i] = x0[i] * ex[:, i] + self.theta[i] * \
                ( 1- ex[:,i]) + self.sigma[i, i] *ex[:, i] * W[:, i] / np.sqrt(2 * self.kappa[i, i])  
            else : 
                X[:, i] = x0[i]  + W[:, i] 
        return X        


"""
Solver of the matrix Riccati equation for the matrix A (Section 3)
"""
class Asolver : 
    n = None
    delta = None
    kappa = None
    dt = None
    t = None
    corr_matrix = None
    corr_matrix_inv = None
    kCk_term = None
    """
    n - dimension
    delta - distortion rate (1 /(1 - \gamma)), \gamma - is the risk aversion coefficient
    kappa - a matrix of reversion speeds. 
    dt - time step
    t - a grid for time variable [0, dt, 2*dt, ... , T].
    corr_matrix - a correlation matrix of n-dimensional Wiener process.
    corr_matrix_inv - inverse to corr_matrix matrix. 
    kCk_term =  \frac{\delta(\delta -1)}{2} \kappa \Corrmatrix^{-1} \kappa (see Section 3)
    """
    
    def __init__(self, OU, gamma, T, dt) :
        self.n = OU.n
        self.dt = dt
        self.t = np.arange(0, T, dt)
        self.delta = 1 / (1- gamma)
        self.kappa = OU.kappa
        self.corr_matrix = OU.corr_matrix
        self.corr_matrix_inv = np.linalg.inv(self.corr_matrix);
        tmp = np.dot(self.kappa, self.corr_matrix_inv)
        tmp = np.dot(tmp, self.kappa)
        self.kCk_term = tmp * (self.delta - 1) * self.delta / 2

    """
    Calculate a matrix of derivative value at time t.
    """
    def _dAdt(self, A, t, params = None) : 
        n = self.n
        A = A[0:len(A) - 1]
    #    A_matrix = vec2symmetric(A,n)
        A_AT = np.reshape(A, (n, n))
        A_AT = A_AT + A_AT.T
    #    derivs = 0.5 * np.dot(np.dot(A_AT, CorrMat), A_AT) - delta * np.dot(kappamat, A_AT) + LastTerm
        derivs = 0.5 * np.dot(np.dot(A_AT, self.corr_matrix), A_AT) +\
                - 0.5 * (self.delta + 1 )* np.dot(self.kappa, A_AT)  +  \
                - 0.5 * (self.delta - 1 )* np.dot(A_AT, self.kappa)  +  \
                + self.kCk_term
        derivs = np.reshape(derivs, n*n);
        B_der = np.zeros(1)
        for i in range(n) : 
            for j in range(n) : 
                B_der = B_der + (A_AT[i, j]) * self.corr_matrix[i,j]
        B_der = 0.5 * B_der
        return np.concatenate((derivs, B_der))        


    """
    solver 
    Output a matrix (n * n + 1) \times (T / dt)
    We return function of the DIRECT time (not inverse)
    time slice matrices (n \times n) are converted to the vectors of the length n * n.
    The last element is the trace term (see section 3)
    """

    def solve(self) :
    #    A0 = np.zeros(int(n * (n +1) / 2 + 1))
        n = self.n
        A0 = np.zeros(int(n ** 2 + 1));
        psoln = odeint(self._dAdt, A0, self.t, args = (None, )) 
        psoln = np.flip(psoln, 0)
        return psoln

#    def getAB0(self) : 
#        AB = self.solve()
#        return [np.reshape(AB[0, 0 : AB.shape[1] - 1], (self.n, self. n)), AB[0, -1]]
#
#    def perturbedKappaSolve(self) : 
#        kappa_vec = np.diag(self.kappa)
#        k = np.min(kappa_vec)
#        kappa_vec = kappa_vec - k;
#        d_kappa = np.diag(kappa_vec)
#        dk_Corr = np.dot(self.corr_matrix_inv, d_kappa) + np.dot(d_kappa, self.corr_matrix_inv)
#        answer = np.zeros((len(self.t), int(self.n ** 2)))
#        sq_delta = np.sqrt(self.delta)
#        omega = (1 - sq_delta) / (1 + sq_delta)
#        for i in range (answer.shape[0]) : 
#            tau = self.t[i]
#            exp_val = np.exp( 2. * k * sq_delta * tau)
#            Psi = k * sq_delta * (sq_delta - 1) * (exp_val  -  1) / (exp_val + omega) / 2
#            coef = 1 / 2 / k + sq_delta * tau * (omega / (exp_val + omega))
#            last_term = sq_delta * tau *  0.5 *  k * sq_delta * (sq_delta - 1) / (exp_val + omega)
#            answer[i, :] = np.reshape(Psi *(self.corr_matrix_inv + coef * dk_Corr) + dk_Corr * last_term, self.n ** 2)
#        return  np.flip(answer, 0)

"""
Solver of the matrix Riccati equation for the matrix Q (Section 6).
This routine is used to analyse effect on parametrs mis-specification
 and to compute the moments of the terminal wealth.
"""
class AQsolver (Asolver) :
    """
    kappa_true - a diagonal matrix of true value of reversion speeds 
    corr_matrix_true- a matrix of true correlations.
    sigma_ratio  =  \sigma * \sigma_est^{-1}, \sigma - a diagonal matrix of true noise magnitudes
     \sigma_est - a diagonal matrix of estimated noise magnitudes
    """
    kappa_true = None
    corr_matrix_true = None
    sigma_ratio = None
    def __init__(self, OU_true, OU_ms, gamma, epsilon, T, dt) : 
        super().__init__(OU_ms, gamma, T, dt)
        self.epsilon = epsilon
        self.kappa_true = OU_true.kappa
        self.corr_matrix_true = OU_true.corr_matrix
        self.sigma_ratio = np.dot(np.linalg.inv(OU_ms.sigma), OU_true.sigma)
    """
    Calculate of derivatives dA/dt and dQ/dt at time t.
    """    
    def _dAQdt(self, AQ, t, params= None) : 
        n = self.n
        dA = super()._dAdt(AQ[0: n * n + 1], self.t)
        A = AQ[0 : n * n]
        A = np.reshape(A, (n,n))
        beta = -self.delta * np.dot(self.corr_matrix_inv, self.kappa) + A + A.T;
        beta = np.dot(self.sigma_ratio, beta)
        beta = np.dot(beta, self.sigma_ratio)
        Q_QT = AQ[n * n + 1 : len(AQ) - 1];
        Q_QT = np.reshape(Q_QT, (n,n))
        Q_QT = Q_QT + Q_QT.T
        betaT_C = np.dot(beta.T, self.corr_matrix_true) * self.epsilon
        dQ = 0.5 * np.dot(np.dot(Q_QT, self.corr_matrix_true), Q_QT) + \
        + np.dot(betaT_C - self.kappa_true, Q_QT) \
        + 0.5 * (self.epsilon - 1) * np.dot(betaT_C, beta) \
        -self.epsilon * np.dot(beta.T, self.kappa_true);
        dQ = np.reshape(dQ, n*n)
        P_der = np.zeros(1)
        for i in range (n):
            for j in range(n) : 
                P_der = P_der +Q_QT[i,j] * self.corr_matrix_true[i,j]
        P_der = 0.5 * P_der
        return np.concatenate((dA, dQ, P_der))


    """
    solver 
    Output a matrix (2 * n * n) \times (T / dt)
    A - first, Q -last
    We return functions of the DIRECT time (not inverse)
    time slice matrices (n \times n) are converted to the vectors of the length n * n.  
    """


    def solve(self) :
    #    A0 = np.zeros(int(n * (n +1) / 2 + 1))
        AQ0 = np.zeros(int(2 * (self.n ** 2 + 1)));
        psoln = odeint(self._dAQdt, AQ0, self.t, args = (None, )) 
        psoln = np.flip(psoln, 0)
        return psoln
    """
    Calculate the value of Q at direct time t = 0. 
    """
    def getQP0(self) : 
        AQ = self.solve()
        return [np.reshape(AQ[0, self.n ** 2 + 1: AQ.shape[1] - 1], (self.n, self.n)), AQ[0, -1]]

"""
Сlass corresponded to the wealth process generated by the optimal strategy
OU - n-dimensional OU process
gamma - risk aversion
delta - distortion rate 1 / (1 - gamma)
T- time horizon
dt - time step
A_t - matrix A
B - trace term (see Section 3)
"""
class W_t:
    OU = None
    corr_matrix_inv = None
    gamma = None
    delta = None
    T = None
    dt = None
    A_t = None
    B = None
    def __init__(self, gamma, OU, T, dt) : 
        self.T = T
        self.dt = dt
        self.gamma = gamma
        self.delta  = 1 / (1 - gamma)
        self.OU = OU
        Asol = Asolver(OU, gamma, T, dt)
        self.corr_matrix_inv = Asol.corr_matrix_inv
        A = Asol.solve()
        self.A_t = A[:, 0 : A.shape[1] - 1]
        self.B = A[:, -1]
    """
    Calculate the value of optimal control for a given values W_t, X_t and t,
    i - is the number of asset.
    """    
    def alpha_star(self, W_t, X_t, i) :
        A_t = np.reshape(self.A_t[i, :], (self.OU.n, self.OU.n)) 
        y = np.reshape(X_t, self.OU.theta.shape) - self.OU.theta
        inner_matrix = -self.delta * np.dot(self.corr_matrix_inv, self.OU.kappa) + (A_t + A_t.T)
        return W_t * np.dot(np.dot(self.OU.sigma_inv, inner_matrix), np.dot(self.OU.sigma_inv, y))

    """
    Simulation
    Output: tuple of trajectories : 
        Wealth
        Spread dynamics
        Positions
    """        
    def simulate (self, w0, x0) :
        X_t = self.OU.simulate(x0, self.T, self.dt)  
        sample_sz = X_t.shape[0]
        W = np.zeros((sample_sz, 1));
        positions_vec = np.zeros((sample_sz, self.OU.n))
        W = np.zeros((sample_sz, 1))
        W[0] = w0
        for i in range(sample_sz - 1) : 
            positions_vec[i, :] = self.alpha_star(W[i], X_t[i, :], i).T
            dX_t = np.reshape(X_t[i+1,:] - X_t[i, :], (1, self.OU.n))
            W[i + 1] = W[i] + np.dot(dX_t, positions_vec[i])
        return [W, X_t, positions_vec]    


    """
    Calculation of the moments of the terminal wealth distribution
    """        
    def momentWT(self, w0, x0, j) : 
        AQsol = AQsolver(self.OU, self.OU, self.gamma, j, self.T, self.dt)
        [Q, P] = AQsol.getQP0()
        x0 = np.reshape(x0, (self.OU.n, 1))
        y = x0 - self.OU.theta
        exp_arg = np.dot(y.T, self.OU.sigma_inv);        
        exp_arg = np.dot(exp_arg, Q);
        exp_arg = np.dot(exp_arg, self.OU.sigma_inv);
        exp_arg = np.dot(exp_arg, y);
        exp_arg = exp_arg + P;
        exp_arg = np.asscalar(exp_arg)
        return float(w0) ** j * np.exp(exp_arg)     

    """
    Calculation of the expectation of the terminal wealth  
    """            
    def EWT(self, w0, x0) : 
        return self.momentWT(w0, x0, 1)

    """
    Calculation of the variance of the terminal wealth  
    """                
    def varWT(self, w0, x0) : 
        return self.momentWT(w0, x0, 2) - self.momentWT(w0, x0, 1) ** 2


    """
    Calculation of first two centered moments and Sharpe ratio  
    """                
    def moments_and_Sharpe(self) : 
        EWT= self.EWT
        varWT = self.varWT
        return [EWT, varWT, EWT / np.sqrt(varWT)]


    """
    Calculation of the value of the value function J
    """                        
    def J_value(self, w0, x0) :
        return self.momentWT(w0, x0, self.gamma) / self.gamma

    """
    Calculation of matrix of the position size multipliers
    """
    def D(self) :
        D = self.A_t.copy()
        myop_dem = self.delta * np.dot(self.corr_matrix_inv, self.OU.kappa)
        for i in range(D.shape[0]) :
            A = np.reshape(D[i, :], (self.OU.n, self.OU.n))
            A = myop_dem -(A + A.T)
            D[i,:] = np.reshape(A, (1, self.OU.n ** 2))
        return D    


    """
    Visualization of the 1D strategy performance
    """
    def plot_simulation1D(self, w0, x0) : 
        [W, X, pos] = self.simulate(w0, x0)
        ew = self.EWT(w0, x0)
        stdWT = np.sqrt(self.varWT(w0, x0))
        stdLine = [ew - stdWT, ew + stdWT]
        t = np.arange(0, self.T, self.dt)
        linewidth = 2
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(t, W, linewidth = linewidth)   
#        plt.xlabel('t')
        plt.ylabel(r'$W_t$')
        ewplt= plt.scatter(t[-1], ew)
        varplt = plt.scatter([t[-1], t[-1]], stdLine)
        plt.legend((ewplt, varplt), ('$\mathbb{E}[W_T]$', '$\mathbb{E}[W_T] \pm \sqrt{Var[W_T]}$'), scatterpoints=1, loc='best')
        plt.title('Wealth dynamics,   '  + '$\gamma$  = ' + str(self.gamma))
        plt.grid(True)
        
        
        D = self.D()
        
        plt.subplot(2, 1, 2)
        plt.xlabel('t')
 #       plt.ylabel('$X_t$')
        plt.title('Prices dynamics')
        plt.grid(True)
        plt.plot(t, X[:, 0], linewidth = linewidth)
        plt.plot(t,pos[:, 0], linewidth = linewidth)

        plt.plot(t,  1 / np.sqrt(D[:, 0]), linewidth = linewidth, color = 'r')
        plt.plot(t,  -1 / np.sqrt(D[:, 0]), linewidth = linewidth, color = 'r')

        plt.legend(['$X_t$', '$\\alpha_t$', '$\pm 1/\sqrt{D(T-t)}$'], loc = 'best')


    """
    Visualization of the multidimensional strategy performance
    """
    def plot_simulation(self, w0, x0) : 
        [W, X, pos] = self.simulate(w0, x0)
        ew = self.EWT(w0, x0)
        stdWT = np.sqrt(self.varWT(w0, x0))
        stdLine = [ew - stdWT, ew + stdWT]
        t = np.arange(0, self.T, self.dt)
        

        KSTHX0 = np.zeros((4, len(x0)))
        KSTHX0[0, :] = np.diag(self.OU.kappa)
        KSTHX0[1, :] = np.diag(self.OU.sigma)
        KSTHX0[2, :] = self.OU.theta.T
        KSTHX0[3, :] = x0
   
        precision = '%.3f'
#        linewidth = 2 / np.sqrt(self.OU.n)
        
        linewidth = 1.2
        if self.OU.n  > 4 : 
            linewidth = 0.7
        
        if (self.OU.n  > 4) : 
            precision = '%.2f'
        if (self.OU.n  > 8) :
            precision = '%.1f'
            
        plt.figure()
        plt.subplot(2, 3, 1)
        plt.plot(t, W, linewidth = linewidth)   
#        plt.xlabel('t')
        plt.ylabel(r'$W_t$')
        ewplt= plt.scatter(t[-1], ew)
        varplt = plt.scatter([t[-1], t[-1]], stdLine)
        plt.legend((ewplt, varplt), ('$E[W_T]$', '$E[W_T] \pm \sqrt{Var[W_T]}$'), scatterpoints=1, loc='best')
        plt.title('Wealth dynamics,   '  + '$\gamma$  = ' + str(self.gamma))
        plt.grid(True)
         
        p = plt.subplot(2, 3, 2)
#        plt.xlabel('t')
 #       plt.ylabel('$X_t$')
        plt.title('Prices dynamics')
        cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
        plt.grid(True)
        for i in range(0, X.shape[1]) :
            plt.plot(t, X[:, i], linewidth = linewidth)
        if (self.OU.n <5) :     
            plt.legend(np.arange(self.OU.n) + 1, loc = 'best')


        ax = plt.subplot(2, 3, 3)
        extent = (0, KSTHX0.shape[1], KSTHX0.shape[0], 0)
        plt.imshow(KSTHX0, cmap = 'PuOr', extent = extent)
        plt.clim([0, np.max(KSTHX0)])
        plt.colorbar()
        plt.title('Process parameters')
        nots = ['$\kappa$', '$\sigma$', '$\\theta$', '$x_0$'];
        if (self.OU.n < 30) :
            plt.grid(color = 'k')
        ax.set_yticks(np.arange(0., 4., 1))
        ax.set_yticklabels(nots)
        ax.set_xticks(np.arange(0, self.OU.n))
        ax.set_xticklabels(np.arange(0, self.OU.n) + 1)
#            ax.set_xticks(np.arange(4)
        ch_col = 0.8 * np.amax(KSTHX0)
        if (self.OU.n  < 8) : 
            for i in range(KSTHX0.shape[0]) : 
                for j in range(KSTHX0.shape[1]) :
                    cl_key = 'k' 
                    if KSTHX0[i, j] > ch_col : 
                        cl_key = 'w'
                    ax.text(j,i, '%.2f'%KSTHX0[i, j], ha = "left", va = "top", color=cl_key)

#        myop_dem = np.reshape(-self.delta * np.dot(self.corr_matrix_inv, self.OU.kappa), (self.OU.n ** 2 , 1))
        plt.subplot(2, 3, 4) 
        plt.grid(True)
        plt.xlabel('t')
        plt.ylabel('$D(T-t)$')

#        plt.title('A matrix')
#        for i in range (0, self.A_t.shape[1]) :
#            plt.plot(t,  self.A_t[:, i], linewidth = linewidth)
#  #      plt.plot(t, self.B, linewidth = linewidth)    
        D = self.D()
        plt.title('Position size multiplier matrix')
        if (self.OU.n == 2) : 
            for i in range (0, D.shape[1]) :
                plt.plot(t,  D[:, i], linewidth = linewidth)
                plt.legend(['$D_{11}$', '$D_{12}$', '$D_{21}$', '$D_{22}$'], loc ='best') 
        else :
            for i in range (0, D.shape[1]) :
                plt.plot(t,  D[:, i], linewidth = linewidth, color = cols[i % self.OU.n])
      #          plt.plot(t,  D[:, i], linewidth = linewidth)
  #      plt.plot(t, self.B, linewidth = linewidth)    
        
        plt.subplot(2, 3, 5)
        plt.grid(True)
        plt.xlabel('t')
    #    plt.ylabel('$\\alpha_t$')
        plt.title('Positions')
        for i in range (0, pos.shape[1]) :
            plt.plot(t,pos[:, i], linewidth = linewidth)

        ax = plt.subplot(2, 3, 6)
        plt.title('Correlation matrix  $\\Theta$')
        extent = (0, self.OU.corr_matrix.shape[1], self.OU.corr_matrix.shape[0], 0)
        plt.imshow(self.OU.corr_matrix, cmap = 'PuOr', extent = extent)
        ax.set_xticks(np.arange(0, self.OU.n))
        ax.set_xticklabels(np.arange(1, 1+self.OU.n))
        ax.set_yticks(np.arange(0, self.OU.n))
        ax.set_yticklabels(np.arange(1, 1+self.OU.n))
        if (self.OU.n < 30) :
            plt.grid(color = 'k')
        ax.set_frame_on(False)
        plt.clim([-1, 1])
        plt.colorbar()

        if (self.OU.n  < 8) : 
            
            for i in range(self.OU.corr_matrix.shape[0]) : 
                for j in range(self.OU.corr_matrix.shape[1]) :
                    cl_key = 'k'
                    if (self.OU.corr_matrix[i, j] > 0.7) : 
                        cl_key = 'w'
                    ax.text(j,i, '%.2f'%self.OU.corr_matrix[i, j], ha = "left", va = "top", color=cl_key)
 


"""
Сlass corresponded to the wealth process generated by the strategy with MISSPECIFIED parameters
derived from the base class W_t
OU_ms - Ornstein--Uhlenbeck process with Estimated parameters 
        (used only as the data structure, we do not performed any simulations of this process 
ms_corr_matrix_inv - the inverse to the estimated correaltion matrix
ms_A_t - the solution  of Riccati equation with estimated parameters (\hat{A} see Section 6)
ms_B - the term corresponded to the trace      
"""
class ms_W_t(W_t) :
    OU_ms = None
    ms_corr_matrix_inv = None
    ms_A_t = None
    ms_B = None    
    def __init__(self, gamma, OU, OU_ms, T, dt) :         
        super().__init__(gamma, OU, T, dt)
        self.OU_ms = OU_ms 
        Asol = Asolver(self.OU_ms, gamma, T, dt)
        self.ms_corr_matrix_inv = Asol.corr_matrix_inv
        A = Asol.solve()
        self.ms_A_t = A[:, 0 : A.shape[1] - 1]
        self.ms_B = A[:, -1]


    """
    trading strategy on the estimated parameters. 
    """    
    def alpha_star(self, W_t, W_t_ms, X_t, i) :
        A_t = np.reshape(self.A_t[i, :], (self.OU.n, self.OU.n)) 
        ms_A_t = np.reshape(self.ms_A_t[i, :], (self.OU.n, self.OU.n))
        y = np.reshape(X_t, self.OU.theta.shape) - self.OU.theta
        inner_matrix = -self.delta * np.dot(self.corr_matrix_inv, self.OU.kappa) + (A_t + A_t.T)
        inner_matrix_ms = -self.delta * np.dot(self.ms_corr_matrix_inv, self.OU_ms.kappa) + (ms_A_t + ms_A_t.T)
        
        return [W_t * np.dot(np.dot(self.OU.sigma_inv, inner_matrix), np.dot(self.OU.sigma_inv, y)).T, \
                W_t_ms * np.dot(np.dot(self.OU_ms.sigma_inv, inner_matrix_ms), np.dot(self.OU_ms.sigma_inv, y)).T]

    """
    Simulation of the wealth corresponded to the estimated parameters.
    """        
    def simulate (self, w0, x0) :
        X_t = self.OU.simulate(x0, self.T, self.dt)  
        sample_sz = X_t.shape[0]
        W = np.zeros((sample_sz, 1));
        true_pos = np.zeros((sample_sz, self.OU.n))
        ms_pos = np.zeros((sample_sz, self.OU.n))
        W = np.zeros((sample_sz, 1))
        W[0] = w0
        W_ms = W.copy()
        for i in range(sample_sz - 1) : 
            [true_pos[i, :], ms_pos[i, :]]= self.alpha_star(W[i], W_ms[i], X_t[i, :], i)
            dX_t = np.reshape(X_t[i+1,:] - X_t[i, :], (1, self.OU.n))
            W[i + 1] = W[i] + np.dot(dX_t, true_pos[i])
            W_ms[i + 1] = W_ms[i] + np.dot(dX_t, ms_pos[i])
        return [W, W_ms, X_t, true_pos, ms_pos]    

    """
    Calculation of expected terminal wealth generated by the strategy via estimated parameters. 
    """        
    def EWTms(self, w0, x0) : 
        return self.momentWTms(w0, x0, 1)

    """
    Calculation of the moments of expected terminal wealth generated by the strategy via estimated parameters. 
    """        
    def momentWTms(self, w0, x0, j) : 
        AQsol = AQsolver(self.OU, self.OU_ms, self.gamma, j, self.T, self.dt)
        [Q, P] = AQsol.getQP0()
        x0 = np.reshape(x0, (self.OU.n, 1))
        y = x0 - self.OU.theta
        exp_arg = np.dot(y.T, self.OU.sigma_inv);        
        exp_arg = np.dot(exp_arg, Q);
        exp_arg = np.dot(exp_arg, self.OU.sigma_inv);
        exp_arg = np.dot(exp_arg, y);
        exp_arg = exp_arg + P;
        exp_arg = np.asscalar(exp_arg)
        return float(w0) ** j * np.exp(exp_arg)     

    """
    Difference between values of the values functions (misspecified and true)  
    """
    def ms_comp(self, x0) : 
        return np.abs(self.gamma) * (self.J_value(1, x0) - self.J_value_ms(1, x0))  
    
    
    """
    The value corressponded to the misspecified parameters
    """    
    def J_value_ms(self, w0, x0) :
        return self.momentWTms(w0, x0, self.gamma) / self.gamma

    """
    Calculation of the variance of the terminal wealth (misspecified parameters)
    """
    def varWTms(self, w0, x0) : 
        return self.momentWTms(w0, x0, 2) - self.momentWTms(w0, x0, 1) ** 2
    
    """
    Visualization of the simulation
    """
    def plot_simulation(self, w0, x0) : 
        [W, W_ms, X, true_pos, ms_pos] = self.simulate(w0, x0)
        ew = self.EWT(w0, x0)
        ew_ms = self.EWTms(w0, x0)
        stdWT = np.sqrt(self.varWT(w0, x0))
        stdWT_ms = np.sqrt(self.varWTms(w0, x0))
        stdLine = [ew - stdWT, ew + stdWT]
        stdLine_ms = [ew_ms - stdWT_ms, ew_ms + stdWT_ms]
        
        linewidth = 2 / np.sqrt(self.OU.n)
        t = np.arange(0, self.T, self.dt)
            
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.plot(t, W, linewidth = linewidth)   
        plt.plot(t, W_ms, linewidth = linewidth)   
        plt.scatter(t[-1], ew)
        plt.scatter([t[-1], t[-1]], stdLine)
        plt.scatter(t[-1], ew_ms)
        plt.scatter([t[-1], t[-1]], stdLine_ms)

        plt.title('Wealth process, $\gamma$  = ' + str(self.gamma))
        plt.legend(['W', 'W_ms'])
        plt.grid(True)
         
        plt.subplot(2, 2, 3)
        plt.title('OU processes')
        plt.grid(True)
        for i in range(0, X.shape[1]) :
            plt.plot(t, X[:, i], linewidth = linewidth)
#        plt.legend(np.arange(self.OU.n) + 1, loc = 0)

        
        plt.subplot(2, 2, 2)
        plt.grid(True)
        plt.title('True Positions')
        for i in range (0, true_pos.shape[1]) :
            plt.plot(t, true_pos[:, i], linewidth = linewidth)

        plt.subplot(2, 2, 4)
        plt.grid(True)
        plt.title('Est Positions')
        for i in range (0, true_pos.shape[1]) :
            plt.plot(t, ms_pos[:, i], linewidth = linewidth)


"""
A visualization of the misspecified kappas (2d example)
gamma_range - range for gamma's values
kappa  - true kappas (for first and second assets)
rho_range - range for correaltion's values
kappa_err - the percent of the error in kappa_estimation
dsc_rate - grid_parameter for visualization
mult_dt - multiplicator for dt
T - time horizon
key - key for visualization: 3d - 3d plot, other - heatmap. 
"""
def kappa_misspecified_test2D(gamma_range, kappa, rho_range, kappa_err, dsc_rate, mult_dt, T, key) :
    sigma = np.array([1, 1])
    theta = np.array([0., 0.])
    x0 = theta.copy()
    lev_count = 15
    dt = 1/252 / mult_dt
#    kappa1lg = np.linspace(-1.5, 1.5, dsc_rate)
#    kappa2lg = np.linspace(-1.5, 1.5, dsc_rate)
#    kappams1 = kappa[0] * np.exp(kappa1lg)
#    kappams2 = kappa[1] * np.exp(kappa2lg)
    kappams1 = np.linspace( kappa[0]* (1- kappa_err), kappa[0] * (1 + 3 * kappa_err), dsc_rate)
    kappams2 = np.linspace( kappa[1]* (1- kappa_err), kappa[1] * (1 + 3*kappa_err), dsc_rate)
    X, Y = np.meshgrid(kappams1, kappams2); 
    Xlog , Ylog = np.meshgrid(np.log(kappams1 / kappa[0]), np.log(kappams2 / kappa[1]))
    corr_matrix = np.zeros((2, 2)) + 1 
    ms_test = np.zeros(X.shape)  
    kappa_ms = kappa.copy();
    gamma_len = len(gamma_range)
    rho_len = len(rho_range)
    plt.figure()  
    plt_it = 1    
    sum_count = gamma_len * rho_len * X.shape[0] * Y.shape[1]
    cur_it = 0
    for s in range(gamma_len) : 
        gamma  = gamma_range[s]
        for k in range(rho_len) :
            rho = rho_range[k]
            corr_matrix[0, 1] = rho
            corr_matrix[1, 0] = rho
            OU = OU_process(kappa, sigma, corr_matrix, theta)
            for i in range(X.shape[0]) :
                if (i % 10==0) : 
                    print(cur_it / sum_count)
                for j in range(Y.shape[1]) :  
                    kappa_ms[0] = X[i, j];
                    kappa_ms[1] = Y[i, j];
                    OU_ms = OU_process(kappa_ms, sigma, corr_matrix, theta)
                    W_t = ms_W_t(gamma, OU, OU_ms, T, dt) 
                    ms_test[i, j] = W_t.ms_comp(x0) 
                    cur_it = cur_it + 1
            tttl = r'$\gamma =  '+ str(gamma) + '\quad  ' + '\\rho = ' + str(rho) + '$'
            if key == '3d' :         
                ax = plt.subplot(gamma_len, rho_len, plt_it, projection = '3d')
                ax.plot_surface(Xlog, Ylog, ms_test, cmap = 'PuOr', alpha = 0.6)
     #           ax.plot_wireframe(Xlog, Ylog, ms_test, cmap = 'PuOr')
                ax.contour(Xlog, Ylog, ms_test, np.linspace(0, np.max(ms_test), lev_count), zdir = 'z', offset = 0, cmap = 'PuOr')            
                ax.contour(Xlog, Ylog, ms_test, zdir = 'x', offset = Xlog[0, 0], cmap = 'PuOr') 
                ax.contour(Xlog, Ylog, ms_test, zdir = 'y', offset = Ylog[-1, -1], cmap = 'PuOr')
                ax.set_xlabel('$log (\kappa_1^* / \kappa_1)$')            
                ax.set_ylabel('$log (\kappa_2^* / \kappa_2)$')   
                ax.set_zlabel('$\propto |J^* - J|$')   
                ax.set_zticklabels([])
     #           plt.subplot(gamma_len, rho_len, plt_it)
                ax.scatter(np.array([0]), np.array([0]), np.array([0]), 'k') 
                ax.set_title(tttl)
            else :   
                plt.subplot(gamma_len, rho_len, plt_it)
                if (plt_it == 1) or (plt_it ==  rho_len + 1): 
                    plt.ylabel('$log (\kappa_2^* / \kappa_2)$')   
                if (plt_it >  rho_len) :    
                    plt.xlabel('$log (\kappa_1^* / \kappa_1)$')            
                C1 = plt.contourf(Xlog,Ylog, ms_test, np.linspace(0, np.max(ms_test), 3 * lev_count), cmap = 'PuOr')

                plt.contour(C1, levels=C1.levels[::2], colors='k')
                plt.scatter(0, 0)             
                plt.title(tttl)
            plt_it = plt_it + 1
    
          #  plt.title('fgfg')