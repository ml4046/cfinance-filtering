"""
NOTE: This is deprecated, see kf.py
"""


import numpy as np
from scipy.optimize import fmin
from scipy.stats import norm

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
        P = (1 - K*H)*P_next*(1 - K*H) + K*R*K
    return obj/N

def obj(f, y, args=None):
    """Wrapper for fmin to allow passing target y to obj. function"""
    def fix_y(params):
        if args is not None:
            S0 = args
            return f(y, params, S0)
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
        P = (1 - K*H)*P_next*(1 - K*H) + K*R*K
    return (x_pred, x_update) if return_filter else x_pred

def ekf_heston_obj(y, # list observations
                   params, # list params
                   S0, # float init price
                   N = 1000, # int total timestep
                   dt=1/250 # float J step size, default daily
              ):
    mu = params[0]
    kappa = params[1]
    theta = params[2]
    sigma = params[3]
    rho = params[4]
    v0 = max(1e-3, params[5])
    x0 = np.log(S0)
    N = len(y)
    # init values
    def heston_transition(x):
        x_next = np.matrix([0,0], dtype=np.float64).T
        x_next[0,0] = x[0,0] + (mu-1/2*x[1,0])*dt
        x_next[1,0] = x[1,0] + kappa*(theta-x[1,0])*dt
        return x_next

    x_update = np.matrix([x0, v0]).T
    F = np.matrix([[1, -1/2*dt],
                  [0, 1-kappa*dt]])
    U = np.matrix([[np.sqrt(v0*dt), 0],
                  [0, sigma*np.sqrt(v0*dt)]])
    Q = np.matrix([[1, rho],
                  [rho, 1]])
    H = np.matrix([1,0])
    P = np.matrix([[1e-2, 0],
                  [0, 1e-2]])
    I = np.identity(2)
    obj = 0
    for i in range(1, N):
        x_pred = heston_transition(x_update)
        P_pred = F*P*F.T + U*Q*U.T
        A = H*P_pred*H.T # only have state transition f
        A = A[0,0]
        delta = y[i] - x_pred[0,0]
        delta = delta[0,0]
        obj += np.log(abs(A)) + delta**2/A

        # measurement update
        K = P_pred*H.T/A
        x_update = x_pred + K*delta
        x_update[1,0] = max(1e-5, x_update[1,0]) # ensure vol. is non-neg
        vk = x_update[1,0]
        U = np.matrix([[np.sqrt(vk*dt), 0],
                        [0, sigma*np.sqrt(vk*dt)]])
        P = (I-K*H)*P_pred
    return obj/N

# def is_pos_def(A):
    # return np.all(np.linalg.eigvals(A) > 0)
def is_pos_def(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    return False

def periodic_map(x, c, d):
    """
    Periodic Param mapping provided by Prof. Hirsa
    """
    if ((x>=c) & (x<=d)):
        y = x
    else:
        range = d-c
        n = np.floor((x-c)/range)
        if (n%2 == 0):
            y = x - n*range;
        else:
            y = d + n*range - (x-c)
    return y

def ukf_heston_obj(y, # list observations
                   params, # list params
                   S0, # float init price
                   N=1000, # in total timestep
                   dt=1/250, # float stepsize, default daily
                   return_vals=False
                  ):
    
    mu = periodic_map(params[0], 0.01, 1)
    kappa = periodic_map(params[1], 1, 3)
    theta = periodic_map(params[2], 0.001, 0.2)
    sigma = periodic_map(params[3], 1e-3, 0.7)
    rho = periodic_map(params[4], -1, 1)
    v0 = periodic_map(params[5], 0.01, 0.2) # ensure positive vt
    
    print([mu, kappa, theta, sigma, rho, v0])
    
    y0 = np.log(S0)
    y_hat = y0
    N = len(y)
    # initialize params
    Ew = 0 # expectation of system (gaussian) noise is 0
    Ev = 0 # expectation of measurement (observables)
    P = 1e-5 # P0 covariance is 0
    F = 1 - kappa*dt + 1/2*rho*sigma*dt
    H = -1/2*dt

    # init state variables
    x_pred = 0
    x_update = v0
    x_pred_vals = np.zeros(N)
    x_update_vals = np.zeros(N)
    x_pred_vals[0] = x_update
    y_hat_vals = np.zeros(N) # predicted price
    y_hat_vals[0] = y0
    # init sigma points and weight updates params
    L = 2
    K = 0
    alpha = 1e-3
    beta = 2
    # TODO: weighting schemes
    lda = alpha**2*(L+K) - L
    # lda = 3 - L
    x_sig = np.matrix(np.zeros((L,2*L+1)))
    W_m = np.zeros(2*L+1)
    W_c = np.zeros(2*L+1)
    W_m[0] = lda/(L+lda)
    W_c[0] = lda/(L+lda) + (1-alpha**2+beta)
    W_m[1:] = 1/(2*(L+lda))
    W_c[1:] = 1/(2*(L+lda))

    obj = 0
    eps = 1e-3
    for i in range(1, N):
        # init augmentation process
        x_aug = np.matrix([x_update, Ew]).T
        Q = sigma**2*(1-rho**2)*x_update*dt
        P_aug = np.matrix([[P, 0],
                           [0, Q]])
        
        # make P positive definite
        # replace negative values in diagonal with small constant
        while not is_pos_def(P_aug):
            P_aug = P_aug + eps * np.eye(P_aug.shape[0])
           
        #diag = P_aug[np.diag_indices(P_aug.shape[0])][0]
        #diag[diag<0] = 1e-3
        #P_aug = np.matrix(np.eye(P_aug.shape[0]) * np.array(diag)[0])
        
        u_state = kappa*theta*dt - sigma*rho*mu*dt + sigma*rho*(y[i]-y[i-1])
        
        # generating sigma points
        # note P_aug is diagonal
        rP_aug = np.sqrt(L+lda) * np.sqrt(P_aug)
        x_sig[:, 0] = x_aug
        for k in range(1, L+1):
            x_sig[:, k] = x_aug + rP_aug[:, k-1]
            x_sig[:,L+k] = x_aug - rP_aug[:, k-1]

        # map sigma point through process transition
        F_x_sig = np.array(x_sig[0,:])[0]
        w = np.sqrt(Q) * x_aug[1,0]
        F_x_sig = F * F_x_sig + u_state + np.sqrt(Q)*np.array(x_sig[1,:])[0]

        # get predicted state and cov from mapped sig points
        R = x_pred * dt
        x_pred = max(1e-3, np.sum(W_m * F_x_sig))
        P_pred = np.sum(W_c * (F_x_sig - x_pred)**2) + Q
        x_pred_aug = np.matrix([x_pred, Ev]).T
        P_pred_aug = np.matrix([[P_pred, 0],
                                [0, R]])

        while not is_pos_def(P_pred_aug):
            P_pred_aug = P_pred_aug + eps * np.eye(P_pred_aug.shape[0])
            
        #diag = P_pred_aug[np.diag_indices(P_pred_aug.shape[0])]
        #diag[diag<0] = 1e-3
        #P_pred_aug = np.matrix(np.eye(P_pred_aug.shape[0]) * np.array(diag)[0])
        
        rP_pred_aug = np.sqrt(L+lda) * np.sqrt(P_pred_aug)
        u_obs = y[i-1] + mu*dt # use model predicted y instead?
        x_sig[:, 0] = x_pred_aug

        for k in range(1, L+1):
            x_sig[:, k] = x_pred_aug + rP_pred_aug[:, k-1]
            x_sig[:, L+k] = x_pred_aug + rP_pred_aug[:, k-1]
        H_x_sig = np.array(x_sig[0,:])[0]
        v = np.sqrt(R) * x_pred_aug[1,0]
        H_x_sig = H * H_x_sig + u_obs + np.sqrt(R)* np.array(x_sig[1,:])[0]
        y_hat = np.sum(W_m * H_x_sig)

        # measurement update
        Pyy = np.sum(W_c*(H_x_sig - y_hat)**2) + R
        Pxy = np.sum(W_c*(F_x_sig-x_pred)*(H_x_sig-y_hat))
        K = Pxy/Pyy
        x_update = max(1e-3, x_pred + K*(y[i]-y_hat)) # ensure positive vol
        P = P_pred - K*Pyy*K
        
        # likelihood
        # TODO: check if correct
        delta = y[i] - y_hat
        A = Pyy # don't need H to retrieve entries and R already added
        obj += np.log(np.fabs(A)) + delta**2/A
        # append results
        x_pred_vals[i] = x_pred
        y_hat_vals[i] = y_hat
    return obj/N if not return_vals else (x_pred_vals, y_hat_vals)
