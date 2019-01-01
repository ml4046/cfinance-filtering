import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import scipy.special as spec

from scipy.optimize import fmin, fmin_bfgs
from scipy.stats import norm
from numpy.random import gamma
from filterpy.monte_carlo import systematic_resample, stratified_resample

class PFHeston(object):
    def __init__(self, y, N=1000, dt=1/250, is_log=False):
        self.y = y
        self.logS0 = np.log(y[0]) if not is_log else y[0]
        self.N = N # num particles
        self.dt = dt


    # deprecated: use filter
    def filter_(self, y, params):
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
        weights = np.array([1/self.N] * self.N)
        for i in range(1, len(y)):
            dy = y[i] - y[i-1]
            # prediction
            x_pred = particles + kappa*(theta-particles)*self.dt + \
                    sigma*rho*(dy - (mu-1/2*particles)*self.dt) + \
                    sigma*np.sqrt(particles*(1-rho**2)*self.dt)*norm.rvs()

            # TODO: resample neg. vols
            x_pred = np.maximum(1e-3, x_pred)

            # SIR (update)
            weights = weights * self.likelihood(y[i], x_pred, particles, y[i-1], params)*\
                        self.transition(x_pred, particles, params)/\
                        self.proposal(x_pred, particles, dy, params)
            weights = weights/sum(weights)

            # Resampling
            if self._neff(weights) < 0.3*self.N:
                print('resampling since: {}'.format(self._neff(weights)))
                # x_pred, weights = self._simple_resample(x_pred, weights)
                idxs = systematic_resample(weights)
                x_pred, weights = self._resample_from_index(x_pred, weights, idxs)

            # observation prediction
            # y_hat = y[i-1] + (mu-1/2*x_pred)*self.dt + np.sqrt(particles*self.dt)*norm.rvs()
            # py_hat = np.array([np.mean(self.prediction_density(y_hat[k], y[i-1], x_pred, params)) for k in range(len(y_hat))])
            # py_hat = py_hat/sum(py_hat)
            # y_hat = np.sum(py_hat * y_hat)

            v = max(1e-3, np.average(x_pred, weights=weights))
            particles = x_pred
            # observations[i] = y_hat
            hidden[i] = v
            print('done with step: {}'.format(i))
        return observations, hidden

    def filter(self, params, is_bounds=True, simple_resample=False, predict_obs=False):
        """
        Performs sequential monte-carlo sampling particle filtering
        Note: Currently only supports a bound of parameters
        """
        y = self.y
        N = self.N

        if not is_bounds: # params is an array of param values, not particles
            mu, kappa, theta, sigma, rho, v0 = self._unwrap_params(params)
        else:
            # initialize param states, N particles for each param sampled uniformly
            v0 = params[-1] # params is shape [(lb, ub)_1,...,k, v0]
            params_states = self._init_parameter_states(N, params[:-1])

        observations = np.zeros(len(y))
        hidden = np.zeros(len(y))
        observations[0] = y[0]
        hidden[0] = v0

        # particles = np.maximum(1e-3, self.proposal_sample(self.N, v, dy, params))
        weights = np.array([1/self.N] * self.N)

        # initialize v particles
        particles = norm.rvs(v0, 0.02, N)
        particles = np.maximum(1e-4, particles)

        # storing the estimated parameters each step
        params_steps = np.zeros((len(params)-1, len(y)))
        params_steps.transpose()[0] = np.mean(params_states, axis=1)

        for i in range(1, len(y)):
            dy = y[i] - y[i-1]

            # prediction
            # proposal sample
            x_pred = self.proposal_sample(N, particles, dy, params_states)
            x_pred = np.maximum(1e-3, x_pred)

            # weights
            Li = self.likelihood(y[i], x_pred, particles, y[i-1], params_states)
            I = self.proposal(x_pred, particles, dy, params_states)
            T = self.transition(x_pred, particles, params_states)
            weights = weights * (Li*T/I)
            weights = weights/sum(weights)

            # Resampling
            if self._neff(weights) < 0.7*self.N:
                print('resampling since: {}'.format(self._neff(weights)))
                if simple_resample:
                    x_pred, weights, params_states = self._simple_resample(x_pred, weights, params_states)
                else:
                    x_pred, weights, params_states = self._systematic_resample(x_pred, weights, params_states)

            # observation prediction
            if predict_obs:
                y_hat = self.observation_predict(x_pred, particles, y[i-1], np.mean(params_states[0])) # mu is the 0 index
                observations[i] = y_hat
                print("Done with iter: {}".format(i))

            hidden[i] = np.sum(x_pred * weights)
            particles = x_pred
            params_steps.transpose()[i] = np.sum(np.multiply(params_states, weights[np.newaxis, :]), axis=1)

        return (hidden, params_steps, observations) if predict_obs else (hidden, params_steps)

    def obj_likelihood(self, x, dy_next, params):
        if isinstance(params, list): # params is an array of param values, not particles
            mu, kappa, theta, sigma, rho, v0 = self._unwrap_params(params)
        else:
            mu, kappa, theta, sigma, rho = self._unwrap_param_states(params)
        m = dy_next + (mu-1/2*x)*self.dt
        s = np.sqrt(x*self.dt)
        return norm.pdf(x, m, s)

    def proposal(self, x, x_prev, dy, params):
        if isinstance(params, list): # params is an array of param values, not particles
            mu, kappa, theta, sigma, rho, v0 = self._unwrap_params(params)
        else:
            mu, kappa, theta, sigma, rho = self._unwrap_param_states(params)
        m = x_prev + kappa*(theta-x_prev)*self.dt + sigma*rho*(dy - (mu-1/2*x_prev)*self.dt)
        s = sigma*np.sqrt(x_prev*(1-rho**2)*self.dt)
        return norm.pdf(x, m, s)

    def likelihood(self, y, x, x_prev, y_prev, params):
        if isinstance(params, list): # params is an array of param values, not particles
            mu, kappa, theta, sigma, rho, v0 = self._unwrap_params(params)
        else:
            mu, kappa, theta, sigma, rho = self._unwrap_param_states(params)
        m = y_prev + (mu-1/2*x)*self.dt
        s = np.sqrt(x_prev*self.dt)
        return norm.pdf(y, m ,s)

    def transition(self, x, x_prev, params):
        if isinstance(params, list): # params is an array of param values, not particles
            mu, kappa, theta, sigma, rho, v0 = self._unwrap_params(params)
        else:
            mu, kappa, theta, sigma, rho = self._unwrap_param_states(params)
        m = 1/(1+1/2*sigma*rho*self.dt) * (x_prev + kappa*(theta-x_prev)*self.dt + 1/2*sigma*rho*x_prev*self.dt)
        s = 1/(1+1/2*sigma*rho*self.dt) * sigma * np.sqrt(x_prev*self.dt)
        return norm.pdf(x, m, s)

    def prediction_density(self, y, y_prev, x, mu):
        m = y_prev + (mu-1/2*x)*self.dt
        s = np.sqrt(x*self.dt)
        return norm.pdf(y, m, s)

    def observation_predict(self, x_pred, particles, y_prev, mu):
        y_hat = y_prev + (mu-1/2*x_pred)*self.dt + np.sqrt(particles*self.dt)*norm.rvs()
        py_hat = np.array([np.mean(self.prediction_density(y_hat[k], y_prev, x_pred, mu)) for k in range(len(y_hat))])
        py_hat = py_hat/sum(py_hat)
        return np.sum(py_hat * y_hat)

    def _init_parameter_states(self, N, bounds):
        # initialize param states
        params_states = np.zeros((len(bounds)
            , N))
        b0, b1, b2, b3, b4 = bounds
        params_states[0] = np.random.rand(N)*(b0[1]-b0[0])+b0[0]
        params_states[1] = np.random.rand(N)*(b1[1]-b1[0])+b1[0]
        params_states[2] = np.random.rand(N)*(b2[1]-b2[0])+b2[0]
        params_states[3] = np.random.rand(N)*(b3[1]-b3[0])+b3[0]
        params_states[4] = np.random.rand(N)*(b4[1]-b4[0])+b4[0]
        return params_states

    def proposal_sample(self, N, x_prev, dy, params):
        """
        x_prev is array of particles
        """
        if len(params.shape) < 2: # params is an array of param values, not particles
            mu, kappa, theta, sigma, rho, v0 = self._unwrap_params(params)
        else:
            mu, kappa, theta, sigma, rho = self._unwrap_param_states(params)
        m = x_prev + kappa*(theta-x_prev)*self.dt + sigma*rho*(dy - (mu-1/2*x_prev)*self.dt)
        s = sigma*np.sqrt(x_prev*(1-rho**2)*self.dt)
        return norm.rvs(m, s, N)

    def __simple_resample(self, particles, weights):
        N = len(particles)
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1. # avoid round-off error
        indexes = np.searchsorted(cumulative_sum, np.random.rand(N))

	    # resample according to indexes
        particles[:] = particles[indexes]
        new_weights = np.ones(len(weights)) / len(weights)
        return particles, new_weights

    def _resample_from_index(self, particles, weights, indexes):
        particles[:] = particles[indexes]
        weights[:] = weights[indexes]
        new_weights = np.ones(len(weights)) / len(weights)
        return particles, new_weights

    def _systematic_resample(self, x_pred, weights, params_states):
        idxs = systematic_resample(weights)
        params_states[0], _ = self._resample_from_index(params_states[0], weights, idxs)
        params_states[1], _ = self._resample_from_index(params_states[1], weights, idxs)
        params_states[2], _ = self._resample_from_index(params_states[2], weights, idxs)
        params_states[3], _ = self._resample_from_index(params_states[3], weights, idxs)
        params_states[4], _ = self._resample_from_index(params_states[4], weights, idxs)
        x_pred, weights = self._resample_from_index(x_pred, weights, idxs)
        return x_pred, weights, params_states

    def _simple_resample(self, x_pred, weights, params_states):
        params_states[0], _ = self.__simple_resample(params_states[0], weights)
        params_states[1], _ = self.__simple_resample(params_states[1], weights)
        params_states[2], _ = self.__simple_resample(params_states[2], weights)
        params_states[3], _ = self.__simple_resample(params_states[3], weights)
        params_states[4], _ = self.__simple_resample(params_states[4], weights)
        x_pred, weights = self.__simple_resample(x_pred, weights)
        return x_pred, weights, params_states

    def _neff(self, weights):
        return 1. / np.sum(np.square(weights))

    def _unwrap_param_states(self, params_states):
        mu = params_states[0]
        kappa = params_states[1]
        theta = params_states[2]
        sigma = params_states[3]
        rho = params_states[4]
        return mu, kappa, theta, sigma, rho

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

class PFVGSA(object):
    def __init__(self, N=1000, dt=1/250):
        self.dt = dt
        self.N = N # number of particles

    def filter_arrival(self, y, params, is_bounds=False):
        if not is_bounds: # params is an array of param values, not particles
            mu, kappa, theta, sigma, nu, eta, lda, omega = self._unwrap_params(params)
        else:
            # initialize param states, N particles for each param sampled uniformly
            params_states = self._init_parameter_states(len(params), self.N, params)

        xi = 1/params_states[4] # arrival rates are hidden
        weights = np.ones(self.N)/self.N

        arrival_rates = np.zeros(len(y))
        arrival_rates[0] = np.mean(xi)

        # storing the estimated parameters each step
        params_steps = np.zeros((len(params), len(y)))
        params_steps.transpose()[0] = np.mean(params_states, axis=1)

        for i in range(1, len(y)):
            mu, kappa, theta, sigma, nu, eta, lda, omega = self._unwrap_param_states(params_states)
            xj = xi + kappa*(eta-xi)*self.dt + lda*np.sqrt(xi*self.dt)*norm.rvs(size=self.N)
            if sum(xj<0) > 0:
                print('num neg. arrival rate: {}'.format(sum(xj<0)))
                xj = self.resample(xj, xi, params_states)

            weights = weights * self.likelihood_arrival(xj, params_states)
            weights[np.isnan(weights)] = 0
            weights = weights/sum(weights)

            # Resampling
            if self._neff(weights) < 0.7*self.N:
                print('resampling since: {}'.format(self._neff(weights)))
                params_states, _ = self._systematic_resample_states(params_states, weights)
                xj, weights = self._systematic_resample(xj, weights)

            arrival_rates[i] = np.sum(weights * xj)
            params_steps.transpose()[i] = np.sum(np.multiply(params_states, weights[np.newaxis, :]), axis=1)
            xi = xj
        return arrival_rates, params_steps


    def filter(self, y, params, is_bounds=False):
        if not is_bounds: # params is an array of param values, not particles
            mu, kappa, theta, sigma, nu, eta, lda, omega = self._unwrap_params(params)
        else:
            # initialize param states, N particles for each param sampled uniformly
            params_states = self._init_parameter_states(len(params), self.N, params)

        ai = 1/params_states[4]
        xj = self.sample_vol(ai, self.N)
        weights = np.ones(self.N)/self.N

        vol = np.zeros(len(y))
        vol[0] = np.mean(np.mean(xj, axis=0))
        arrivals = np.zeros(len(y))
        arrivals[0] = np.mean(ai)
        # storing the estimated parameters each step
        params_steps = np.zeros((len(params), len(y)))
        params_steps.transpose()[0] = np.mean(params_states, axis=1)
        for i in range(1, len(y)):
            # transition states
            # print(np.sum(np.mean(xj, axis=0)*weights))
            mu, kappa, theta, sigma, nu, eta, lda, omega = self._unwrap_param_states(params_states)
            aj = ai + kappa*(eta-ai)*self.dt + lda*np.sqrt(ai*self.dt)*norm.rvs(size=self.N)
            if sum(aj<0) > 0:
                print('num neg. arrival rate: {}'.format(sum(aj<0)))
                aj = self.resample(aj, ai, params_states)

            xj = self.sample_vol(aj, self.N)
            # compute unconditional state
            xj_uncond = np.mean(xj, axis=0)

            weights = weights * self.likelihood(y[i], xj_uncond, y[i-1], params_states)
            weights = weights/sum(weights)
            
            # Resampling
            if self._neff(weights) < 0.7*self.N:
                print('resampling since: {}'.format(self._neff(weights)))
                params_states, _ = self._systematic_resample_states(params_states, weights)
                xj_uncond, weights = self._systematic_resample(xj_uncond, weights)
            vol[i] = np.sum(weights*xj_uncond)
            arrivals[i] = np.sum(weights * aj)
            params_steps.transpose()[i] = np.sum(np.multiply(params_states, weights[np.newaxis, :]), axis=1)
            ai = aj
        return vol, arrivals, params_steps

    def likelihood(self, obs, x_uncond, obs_prev, params):
        if isinstance(params, list):
            mu, kappa, theta, sigma, nu, eta, lda, omega = self._unwrap_params(params)
        else:
            mu, kappa, theta, sigma, nu, eta, lda, omega = self._unwrap_param_states(params)
        m = obs_prev + (mu+omega)*self.dt + theta*x_uncond
        s = sigma*np.sqrt(x_uncond)
        return norm.pdf(obs, m, s)

    def sample_vol(self, arrival_rates, num_particles):
        vol = np.zeros((len(arrival_rates), num_particles))
        for i in range(len(arrival_rates)):
            vol[i] = sps.gamma.rvs(arrival_rates[i]*self.dt, size=self.N)
        return vol

    def resample(self, aj, ai, params):
        if isinstance(params, list):
            mu, kappa, theta, sigma, nu, eta, lda, omega = self._unwrap_params(params)
        else:
            mu, kappa, theta, sigma, nu, eta, lda, omega = self._unwrap_param_states(params)
        neg_idxs = np.where(aj<0)[0]
        for i in neg_idxs:
            while aj[i] < 0:
                aj[i] = ai[i] + kappa[i]*(eta[i]-ai[i])*self.dt + lda[i]*np.sqrt(ai[i]*self.dt)*norm.rvs()
        return aj

    def likelihood_arrival(self, arrival_rates, params):
        if isinstance(params, list):
            mu, kappa, theta, sigma, nu, eta, lda, omega = self._unwrap_params(params)
        else:
            mu, kappa, theta, sigma, nu, eta, lda, omega = self._unwrap_param_states(params)
        a = 2*np.exp(theta*arrival_rates/sigma**2)
        b = nu**(self.dt/nu)*np.sqrt(2*np.pi)*sigma*spec.gamma(self.dt/nu)
        c = (arrival_rates**2/(2*sigma**2/nu+theta**2))**(self.dt/(2*nu)-1/4)
        e = 1/sigma**2*np.sqrt(arrival_rates**2*(2*sigma**2/nu+theta**2))
        d = spec.kv(self.dt/nu-1/2, e)
        return a/b*c*d

    def _unwrap_params(self, params):
        mu = params[0]
        kappa = params[1] # mean reversion rate
        theta = params[2]
        sigma = params[3]
        nu = params[4]
        eta = params[5] # long-term rate of time change
        lda = params[6] # time change volatility
        omega = 1/nu*np.log(1-theta*nu-sigma**2*nu/2)
        return mu, kappa, theta, sigma, nu, eta, lda, omega

    def _systematic_resample(self, particles, weights):
        idxs = systematic_resample(weights)
        particles[:] = particles[idxs]
        return particles, np.ones(len(weights))/len(weights)

    def _systematic_resample_states(self, params_states, weights):
        idxs = systematic_resample(weights)
        params_states[0] = params_states[0][idxs]
        params_states[1] = params_states[1][idxs]
        params_states[2] = params_states[2][idxs]
        params_states[3] = params_states[3][idxs]
        params_states[4] = params_states[4][idxs]
        params_states[5] = params_states[5][idxs]
        params_states[6] = params_states[6][idxs]
        return params_states, np.ones(len(weights))/len(weights)

    def _neff(self, weights):
        return 1. / np.sum(np.square(weights))


    def _init_parameter_states(self, num_params, N, bounds):
        # initialize param states
        params_states = np.zeros((num_params, N))
        b0, b1, b2, b3, b4, b5, b6 = bounds
        params_states[0] = np.random.rand(N)*(b0[1]-b0[0])+b0[0]
        params_states[1] = np.random.rand(N)*(b1[1]-b1[0])+b1[0]
        params_states[2] = np.random.rand(N)*(b2[1]-b2[0])+b2[0]
        params_states[3] = np.random.rand(N)*(b3[1]-b3[0])+b3[0]
        params_states[4] = np.random.rand(N)*(b4[1]-b4[0])+b4[0]
        params_states[5] = np.random.rand(N)*(b5[1]-b5[0])+b5[0]
        params_states[6] = np.random.rand(N)*(b6[1]-b6[0])+b6[0]        
        return params_states

    def _unwrap_param_states(self, params_states):
        mu = params_states[0]
        kappa = params_states[1] # mean reversion rate
        theta = params_states[2]
        sigma = params_states[3]
        nu = params_states[4]
        eta = params_states[5] # long-term rate of time change
        lda = params_states[6] # time change volatility
        omega = 1/nu*np.log(1-theta*nu-sigma**2*nu/2)
        return mu, kappa, theta, sigma, nu, eta, lda, omega





















