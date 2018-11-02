import numpy as np

from scipy.optimize import fmin

def heston_obj(s0, dt, params):
    kappa = params["kappa"]
    theta = params["theta"]
    lda = params["lda"]
    rho = params["rho"]
    v0 = params["v0"]


def generate_kalman_example(params, N=1000):
    """Generate example for KF"""
    np.random.seed(254)
    F = params[0]
    H = params[1]
    Q = params[2]
    R = params[3]
    x0 = params[4]
    v0 = params[5]

    x = np.zeros(N+1)
    y = np.zeros(N+1)
    
    # step once
    x[0] = x0
    x[1] = F * x0 + np.random.normal(0, np.sqrt(Q))
    y[1] = H * x[1] + v0
    
    for i in range(2,N+1):
        x[i] = F * x[i-1] + np.random.normal(0, np.sqrt(Q))
        y[i] = H * x[i] + np.random.normal(0, np.sqrt(R))
    return x, y

def kalman_obj(y, # int observations 
               params # list params
              ):
    F = params[0]
    H = params[1]
    Q = params[2]
    R = params[3]
    x0 = params[4]
    v0 = params[5]

    # init values
    obj = 0
    x_update = x0
    P = v0
    N = len(y) - 1
    for i in range(1, len(y)):
        # prediction step and objective
        x_pred = F * x_update
        P_next = F*P*F + Q
        S = H*P_next*H + R

        delta = y[i] - H*x_pred
        obj += delta * (1/S) * delta + np.log(np.abs(S))

        # measurement update
        K = P_next * H * (1/S) # kalman gain
        x_update = x_pred + K * delta
        # P = (1 - K*H)*P_next*(1 - K*H) + K*R*K
        P = (1 - K*H)*P_next
    return obj/N

def obj(f, y):
    """Wrapper for fmin to allow passing target y to obj. function"""
    def fix_y(params):
        return f(y, params)
    return fix_y

def kalman_path(y, params, N=1000, return_filter=False):
    F = params[0]
    H = params[1]
    Q = params[2]
    R = params[3]
    x0 = params[4]
    v0 = params[5]

    # init values
    x_pred = np.zeros(N+1)
    x_update = np.zeros(N+1)
    x_update[0] = x0
    P = v0
    for i in range(1, len(x_pred)):
        # prediction step and objective
        x_pred[i] = F * x_update[i-1]
        P_next = F*P*F + Q
        S = H*P_next*H + R
        delta = y[i] - H*x_pred[i]
        
        # measurement update
        K = P_next * H * (1/S) # kalman gain
        x_update[i] = x_pred[i] + K * delta
        # P = (1 - K*H)*P_next*(1 - K*H) + K*R*K
        P = (1 - K*H)*P_next
    return (x_pred, x_update) if return_filter else x_pred
