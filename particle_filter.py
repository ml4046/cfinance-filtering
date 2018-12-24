import numpy as np
from scipy.optimize import fmin
from scipy.stats import norm
import matplotlib.pyplot as plt

class PFHeston(object):
    def __init__(self, y, N=1000, dt=1/250):
        self.y = y
        self.logS0 = np.log(y[0])
        self.N = N # num particles
        self.dt = dt


    def filter(self, y, params):
        """
        Performs sequential monte-carlo sampling particle filtering
        """
        mu, kappa, theta, sigma, rho, v0 = self._unwrap_params(params)
        x_pred = np.array([v0] * self.N)
        v = v0

        observations = np.zeros(len(y))
        hidden = np.zeros(len(y))
        observations[0] = y[0]
        hidden[0] = v0

        # initialize weights and particles
        dy = y[1]-y[0]
        particles = np.maximum(1e-3, self.proposal_sample(self.N, v, dy, params))
        # particles = np.maximum(1e-3, norm.rvs(0, 1, self.N) * sigma*np.sqrt(x_pred*(1-rho**2)*self.dt) + x_pred)
        weights = np.array([1/self.N] * self.N)
        for i in range(1, len(y)):
            dy = y[i] - y[i-1]
            # prediction
            x_pred = particles + kappa*(theta-particles)*self.dt + \
                    sigma*rho*(dy - (mu-1/2*particles)*self.dt) + \
                    sigma*np.sqrt(particles*(1-rho**2)*self.dt)*norm.rvs()
            x_pred = np.maximum(1e-3, x_pred)

    		# SIR (update)
            weights = weights * self.likelihood(y[i], x_pred, particles, y[i-1], params)*\
                        self.transition(x_pred, particles, params)/\
                        self.proposal(x_pred, particles, dy, params)
            weights = weights/sum(weights)

            # Resampling
            if self._neff(weights) < 3/5*self.N:
                print('resampling since: {}'.format(self._neff(weights)))
                x_pred, weights = self._simple_resample(x_pred, weights)

            # observation prediction
            y_hat = y[i-1] + (mu-1/2*x_pred)*self.dt + np.sqrt(particles*self.dt)*norm.rvs()
            py_hat = np.array([np.mean(self.prediction_density(y_hat[k], y[i-1], x_pred, params)) for k in range(len(y_hat))])
            py_hat = py_hat/sum(py_hat)
            y_hat = np.sum(py_hat * y_hat)

            v = max(1e-3, np.average(x_pred, weights=weights))
            particles = x_pred
            observations[i] = y_hat
            hidden[i] = v
            print('done with step: {}'.format(i))
        return observations, hidden


    def proposal(self, x, x_prev, dy, params):
    	mu, kappa, theta, sigma, rho, v0 = self._unwrap_params(params)
    	m = x_prev + kappa*(theta-x_prev)*self.dt + sigma*rho*(dy - (mu-1/2*x_prev)*self.dt)
    	s = sigma*np.sqrt(x_prev*(1-rho**2)*self.dt)
    	return norm.pdf(x, m, s)

    def likelihood(self, y, x, x_prev, y_prev, params):
        mu, kappa, theta, sigma, rho, v0 = self._unwrap_params(params)
        m = y_prev + (mu-1/2*x)*self.dt
        s = np.sqrt(x_prev*self.dt)
        return norm.pdf(y, m ,s)

    def transition(self, x, x_prev, params):
        mu, kappa, theta, sigma, rho, v0 = self._unwrap_params(params)
        m = 1/(1+1/2*sigma*rho*self.dt) * (x_prev + kappa*(theta-x_prev)*self.dt + 1/2*sigma*rho*x_prev*self.dt)
        s = 1/(1+1/2*sigma*rho*self.dt) * sigma * np.sqrt(x_prev*self.dt)
        return norm.pdf(x, m, s)

    def prediction_density(self, y, y_prev, x, params):
        mu, kappa, theta, sigma, rho, v0 = self._unwrap_params(params)
        m = y_prev + (mu-1/2*x)*self.dt
        s = np.sqrt(x*self.dt)
        return norm.pdf(y, m, s)

    def proposal_sample(self, N, x_prev, dy, params):
        """
        x_prev is array of particles
        """
        mu, kappa, theta, sigma, rho, v0 = self._unwrap_params(params)
        m = x_prev + kappa*(theta-x_prev)*self.dt + sigma*rho*(dy - (mu-1/2*x_prev)*self.dt)
        s = sigma*np.sqrt(x_prev*(1-rho**2)*self.dt)
        return norm.rvs(m, s, N)

    def _simple_resample(self, particles, weights):
        N = len(particles)
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1. # avoid round-off error
        indexes = np.searchsorted(cumulative_sum, np.random.rand(N))

	    # resample according to indexes
        particles[:] = particles[indexes]
        weights.fill(1.0 / N)
        return particles, weights

    def _neff(self, weights):
        return 1. / np.sum(np.square(weights))

    def _unwrap_params(self, params):
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
        mu = periodic_map(params[0], 0.01, 1)
        kappa = periodic_map(params[1], 1, 3)
        theta = periodic_map(params[2], 0.001, 0.2)
        sigma = periodic_map(params[3], 1e-3, 0.7)
        rho = periodic_map(params[4], -1, 1)
        v0 = periodic_map(params[5], 1e-3, 0.2) # ensure positive vt
        return mu, kappa, theta, sigma, rho, v0