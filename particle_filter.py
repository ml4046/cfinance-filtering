import numpy as np
from scipy.optimize import fmin, fmin_bfgs
from scipy.stats import norm
import matplotlib.pyplot as plt
from filterpy.monte_carlo import systematic_resample, stratified_resample

class PFHeston(object):
    def __init__(self, y, N=1000, dt=1/250, is_log=False):
        self.y = y
        self.logS0 = np.log(y[0]) if not is_log else y[0]
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

    def obj(self, params):
        """
        Performs sequential monte-carlo sampling particle filtering
        """
        mu, kappa, theta, sigma, rho, v0 = self._unwrap_params(params)
        y = self.y
        N = self.N
        observations = np.zeros(len(y))
        hidden = np.zeros(len(y))
        observations[0] = y[0]
        hidden[0] = v0

        # initialize weights and particles
        dy = y[1]-y[0]
        # particles = np.maximum(1e-3, self.proposal_sample(self.N, v, dy, params))
        weights = np.array([1/self.N] * self.N)

        # initialize param states
        params_states = np.zeros((len(params)-1, N))
        params_states[0] = np.random.rand(N)*(0.5-0.05)+0.05
        params_states[1] = np.random.rand(N)*(9-1)+1
        params_states[2] = np.random.rand(N)*(0.2-0.05)+0.05
        params_states[3] = np.random.rand(N)*(0.91-0.01)+0.01
        params_states[4] = np.random.rand(N)*(0.5) - 0.5

        # initialize v particles
        particles = norm.rvs(v0, 0.02, N)
        particles = np.maximum(1e-4, particles)
        # particles = np.maximum(1e-4, self.proposal_sample(self.N, v0, dy, params))

        all_params = np.zeros((len(params)-1, len(y)))
        all_params[0][0] = np.sum(params_states[0][:] * weights)
        all_params[1][0] = np.sum(params_states[1][:] * weights)
        all_params[2][0] = np.sum(params_states[2][:] * weights)
        all_params[3][0] = np.sum(params_states[3][:] * weights)
        all_params[4][0] = np.sum(params_states[4][:] * weights)

        # param states every step
        params_steps = np.zeros((len(params)-1, len(y), N))
        params_steps[0][0] = params_states[0]
        params_steps[1][0] = params_states[1]
        params_steps[2][0] = params_states[2]
        params_steps[3][0] = params_states[3]
        params_steps[4][0] = params_states[4]

        for i in range(1, len(y)):
            dy = y[i] - y[i-1]

            # prediction
            mu = params_states[0]
            kappa = params_states[1]
            theta = params_states[2]
            sigma = params_states[3]
            rho = params_states[4]

            # proposal sample
            m = particles + kappa*(theta-particles)*self.dt + \
                    sigma*rho*(dy - (mu-1/2*particles)*self.dt)
            s = sigma*np.sqrt(particles*(1-rho**2)*self.dt)
            x_pred = norm.rvs(m, s, N)
            x_pred = np.maximum(1e-3, x_pred)

            # weights
            mLi = y[i-1] + (mu-1/2*x_pred)*self.dt
            sLi = np.sqrt(particles*self.dt)
            Li = norm.pdf(y[i], mLi, sLi)

            mI = particles + kappa*(theta-particles)*self.dt + sigma*rho*(dy-(mu-1/2*particles)*self.dt)
            sI = sigma*np.sqrt(particles*(1-rho**2)*self.dt)
            I = norm.pdf(x_pred, mI, sI)

            c_ = (1+1/2*sigma*rho*self.dt)
            mT = 1/c_ * (particles+kappa*(theta-particles)*self.dt + 1/2*sigma*rho*particles*self.dt)
            sT = 1/c_ * sigma*np.sqrt(particles*self.dt)
            T = norm.pdf(x_pred, mT, sT)

            # print("importance min: {}".format(min(I)))
            weights = weights * (Li*T/I)
            # print("weight max post: {}".format(max(weights)))
            weights = weights/sum(weights)
            print(self._neff(weights))
            # plt.hist(weights, bins=20, density=True)
            # plt.show()

            # Resampling
            if self._neff(weights) < 0.7*self.N:
                print('resampling since: {}'.format(self._neff(weights)))
                # params_states[0], _ = self._simple_resample(params_states[0], weights)
                # params_states[1], _ = self._simple_resample(params_states[1], weights)
                # params_states[2], _ = self._simple_resample(params_states[2], weights)
                # params_states[3], _ = self._simple_resample(params_states[3], weights)
                # params_states[4], _ = self._simple_resample(params_states[4], weights)
                # x_pred, weights = self._simple_resample(x_pred, weights)

                # systematic resample
                idxs = systematic_resample(weights)
                params_states[0], _ = self._resample_from_index(params_states[0], weights, idxs)
                params_states[1], _ = self._resample_from_index(params_states[1], weights, idxs)
                params_states[2], _ = self._resample_from_index(params_states[2], weights, idxs)
                params_states[3], _ = self._resample_from_index(params_states[3], weights, idxs)
                params_states[4], _ = self._resample_from_index(params_states[4], weights, idxs)
                x_pred, weights = self._resample_from_index(x_pred, weights, idxs)

            # plt.hist(params_states[0], bins=20)
            # plt.show()
            hidden[i] = np.sum(x_pred * weights)
            particles = x_pred

            all_params[0][i] = np.sum(params_states[0] * weights)
            all_params[1][i] = np.sum(params_states[1] * weights)
            all_params[2][i] = np.sum(params_states[2] * weights)
            all_params[3][i] = np.sum(params_states[3] * weights)
            all_params[4][i] = np.sum(params_states[4] * weights)

            params_steps[0][i] = params_states[0]
            params_steps[1][i] = params_states[1]
            params_steps[2][i] = params_states[2]
            params_steps[3][i] = params_states[3]
            params_steps[4][i] = params_states[4]

        return hidden, all_params, params_steps

    def obj_likelihood(self, x, dy_next, params):
        mu, kappa, theta, sigma, rho, v0 = self._unwrap_params(params)
        m = dy_next + (mu-1/2*x)*self.dt
        s = np.sqrt(x*self.dt)
        return norm.pdf(x, m, s)

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
        new_weights = np.ones(len(weights)) / len(weights)
        # weights.fill(1.0 / N)
        return particles, new_weights

    def _resample_from_index(self, particles, weights, indexes):
        particles[:] = particles[indexes]
        weights[:] = weights[indexes]
        new_weights = np.ones(len(weights)) / len(weights)
        # weights.fill(1.0 / len(weights))
        return particles, new_weights

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