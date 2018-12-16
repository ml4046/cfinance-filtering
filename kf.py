import numpy as np
from scipy.optimize import fmin

class EKFHeston(object):
    def __init__(self, y, dt=1/250):
        self.y = y # observations
        self.logS0 = np.log(y[0])
        self.dt = dt # default to daily
              
    def observation_transition(self, x, params):
        """
        Heston observation state transition
        """
        mu, kappa, theta, sigma, rho, v0 = self._unwrap_params(params)
        
        x_next = np.matrix([0,0], dtype=np.float64).T
        x_next[0,0] = x[0,0] + (mu-1/2*x[1,0])*self.dt
        x_next[1,0] = x[1,0] + kappa*(theta-x[1,0])*self.dt
        return x_next
    
    def obj(self, params):
        x_update, F, U, Q, H, P_update, I = self._init_transition(params)
        N = len(self.y)
        obj = 0
        for i in range(1, N):
            # time update
            x_pred, P_pred = self._time_update(self.y[i], params, x_update, F, H, 
                                               P_update, U, Q)
            A = H*P_pred*H.T
            A = A[0,0]
            # calc. error
            delta = self.y[i] - x_pred[0,0]
            obj += np.log(abs(A)) + delta**2/A
            # measurement update
            x_update, P_update = self._measurement_update(params, x_pred, P_pred, 
                                                          H, A, delta, I)
        return obj/N
    
    def optimize(self, init_params, maxiter=10000):
        """
        Performs simplex optimization for parameter estimation
        """
        self.num_iter = 1
        def callbackF(xi):
            global arg
            print('i: ' + str(self.num_iter))
            print('x_i: ' + str(xi))
            print('f_i: ' + str(self.obj(xi)))
            self.num_iter += 1
            
        xopt, fopt, _, _, _ = fmin(self.obj, init_params, 
                                   maxiter=maxiter, callback=callbackF, 
                                   disp=True, retall=False, full_output=True)        
        
        
        
    def _time_update(self, y, params, x, F, H, P, U, Q):
        x_pred = self.observation_transition(x, params)
        P_pred = F*P*F.T + U*Q*U.T
        return x_pred, P_pred
    
    def _measurement_update(self, params, x_pred, P_pred, H, A, delta, I):
        mu, kappa, theta, sigma, rho, v0 = self._unwrap_params(params)
        
        K = P_pred*H.T/A
        x_update = x_pred + K*delta
        x_update[1,0] = max(1e-5, x_update[1,0]) # ensure vol. is non-neg
        vk = x_update[1,0]
        U = np.matrix([[np.sqrt(vk*self.dt), 0],
                        [0, sigma*np.sqrt(vk*self.dt)]])
        P_update = (I-K*H)*P_pred
        return x_update, P_update
    
    def _init_transition(self, params):
        """
        Initializes required transition matrices for Kalman Filter
        """
        # params
        mu, kappa, theta, sigma, rho, v0 = self._unwrap_params(params)
        
        x_update = np.matrix([self.logS0, v0]).T
        F = np.matrix([[1, -1/2*self.dt],
                      [0, 1-kappa*self.dt]])
        U = np.matrix([[np.sqrt(v0*self.dt), 0],
                      [0, sigma*np.sqrt(v0*self.dt)]])
        Q = np.matrix([[1, rho],
                      [rho, 1]])
        H = np.matrix([1,0])
        P_update = np.matrix([[1e-2, 0],
                      [0, 1e-2]])
        I = np.identity(2)
        return x_update, F, U, Q, H, P_update, I
    
    def _unwrap_params(self, params):
        mu = params[0]
        kappa = params[1]
        theta = params[2]
        sigma = params[3]
        rho = params[4]
        v0 = max(1e-3, params[5])
        return mu, kappa, theta, sigma, rho, v0
    