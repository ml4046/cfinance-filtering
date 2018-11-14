import numpy as np

from scipy.stats import norm

def simulate_heston(params, S0, r, q, T, N=1000, eps=1e-8):
    kappa = params[0]
    theta = params[1]
    sigma = params[2]
    rho = params[3]
    v0 = params[4]

    # init variables
    dt = T/N
    log_Ss = np.log(S0)
    all_logS = [log_Ss]
    vs = v0

    for i in range(N):
        zv = norm.rvs()
        z = norm.rvs()
        zs = rho*zv + np.sqrt(1-rho**2)*z
        
        vt = vs + kappa*(theta-max(vs, eps))*dt + \
                sigma*np.sqrt(max(vs,eps))*np.sqrt(dt)*zv
        log_St = log_Ss + (r-(1/2)*max(vs, eps))*dt + \
                np.sqrt(max(vs,eps))*np.sqrt(dt)*zs
        all_logS.append(log_St)
        
        # update process
        log_Ss = log_St
        vs = vt
    return all_logS

def simulate_heston_state(params, # list params
                          S0, # float init price
                          r, # init mu
                          N = 1000, # total num steps
                          dt = 1/250 # float step size, default daily
                         ):
    kappa = params[0]
    theta = params[1]
    sigma = params[2]
    rho = params[3]
    v0 = params[4]
    Q = np.matrix([[1, rho],
                   [rho, 1]])
    y = np.matrix(np.zeros((2, N+1)))
    y[0,0] = np.log(S0)
    y[1,0] = v0
    for i in range(1, N+1):
        z1, z2 = np.random.multivariate_normal([0,0], Q)
        v = max(y[1,i-1], 0)
        y[0, i] = y[0, i-1] + (r-1/2*v)*dt + np.sqrt(v*dt)*z1
        y[1, i] = y[1, i-1] + kappa*(theta-v)*dt + sigma*np.sqrt(v*dt)*z2
    return y
