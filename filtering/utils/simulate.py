import numpy as np

from scipy.stats import norm
from numpy.random import gamma

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
                          N = 1000, # total num steps
                          dt = 1/250 # float step size, default daily
                         ):
    mu = params[0]
    kappa = params[1]
    theta = params[2]
    sigma = params[3]
    rho = params[4]
    v0 = params[5]
    Q = np.matrix([[1, rho],
                   [rho, 1]])
    y = np.matrix(np.zeros((2, N+1)))
    y[0,0] = np.log(S0)
    y[1,0] = v0
    for i in range(1, N+1):
        z1, z2 = np.random.multivariate_normal([0,0], Q)
        v = max(y[1,i-1], 0)
        y[0, i] = y[0, i-1] + (mu-1/2*v)*dt + np.sqrt(v*dt)*z1
        y[1, i] = y[1, i-1] + kappa*(theta-v)*dt + sigma*np.sqrt(v*dt)*z2
    return y

def simulate_vg(params, S0, N=1000, dt=1/250):
    mu = params[0]
    theta = params[1]
    sigma = params[2]
    nu = params[3]

    h = dt
    omega = 1/nu * np.log(1-theta*nu-sigma**2*nu/2) # dt*Psi(-u)
    logs = np.zeros(N+1)
    jumps = np.zeros(N+1)

    logs[0] = np.log(S0)
    si = logs[0]
    yi = 1/nu
    for i in range(1, N+1):
        g = gamma(h/nu, nu)
        z = norm.rvs()
        X = theta*g + sigma*np.sqrt(g)*z
        si = si + mu*h + omega*h + X
        logs[i] = si
        jumps[i] = X
    jumps[0] = jumps[1]
    return logs, jumps

def simulate_vgsa(params, S0, N=1000, dt=1/250):

    mu = params[0]
    kappa = params[1] # mean reversion rate
    theta = params[2]
    sigma = params[3]
    nu = params[4]
    eta = params[5] # long-term rate of time change
    lda = params[6] # time change volatility

    def coth(z):
        return (np.exp(2*z) + 1) / (np.exp(2*z) - 1)

    def Psi(u):
        return -1/nu*np.log(1-1j*u*theta*nu+sigma**2*nu*u**2/2)

    def log_phi(u, t, y0):
        gamma = np.sqrt(kappa**2 - 2*lda**2*1j*u)
        log_A = (kappa**2*eta*t/lda**2) - (2*kappa*eta/lda**2) *\
                np.log(np.cosh(gamma*t/2)+(kappa/gamma)*np.sinh(gamma*t/2))
        B = 2*1j*u/(kappa+gamma*coth(gamma*t/2))
        return log_A + B*y0

    logs = np.zeros(N+1)
    jumps = np.zeros(N+1)

    logs[0] = np.log(S0)
    si = logs[0]
    yi = 1/nu
    arrival_rates = np.zeros(N+1)
    arrival_rates[0] = 1/yi
    for i in range(1, N+1):
        z = norm.rvs()
        yj = yi + kappa*(eta-yi)*dt + lda*np.sqrt(yi*dt)*z + lda**2/4*dt*(z**2-1)
        yj = max(1e-5, yi)
        t = dt/2*(yj+yi)
        g = gamma(t/nu, nu)
        z = norm.rvs()
        X = theta*g + sigma*np.sqrt(g)*z
        t1 = i if i == 1 else i-1
        t2 = i
        omega = log_phi(-1j*Psi(-1j), t1*dt, 1/nu) - \
                    log_phi(-1j*Psi(-1j), t2*dt, 1/nu)
        sj = si + mu*dt + omega + X
        logs[i] = sj
        jumps[i] = X
        arrival_rates[i] = 1/yj
        si = sj
        yi = yj

    jumps[0] = jumps[1]
    return logs, jumps















