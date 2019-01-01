import numpy as np
from scipy.optimize import fmin

class EKFHeston(object):
    def __init__(self, y, dt=1/250, is_log=False, bounds=None):
        self.y = y # observations
        self.logS0 = np.log(y[0]) if not is_log else y[0]
        self.dt = dt # default to daily

        # bounds for periodic map
        if bounds is not None:
            self.bounds = bounds 
        else:
            mu = (0.01, 1)
            kappa = (1, 3)
            theta = (1e-3, 0.2)
            sigma = (1e-3, 0.7)
            rho = (-1, 1)
            v0 = (1e-3, 0.2)
            self.bounds = [mu, kappa, theta, sigma, rho, v0]

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
            x_pred, P_pred = self._time_update(params, x_update, F, H, 
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
    
    def filter(self, y, params):
        x_update, F, U, Q, H, P_update, I = self._init_transition(params)
        mu, kappa, theta, sigma, rho, v0 = self._unwrap_params(params)
        print("params: {}".format([mu, kappa, theta, sigma, rho, v0]))
        N = len(y)
        observations = np.zeros(N)
        hidden = np.zeros(N)
        for i in range(1, N):
            # time update
            x_pred, P_pred = self._time_update(params, x_update, F, H, 
                                               P_update, U, Q)
            A = H*P_pred*H.T
            A = A[0,0]
            # calc. error
            delta = y[i] - x_pred[0,0]
            
            # measurement update
            x_update, P_update = self._measurement_update(params, x_pred, P_pred, 
                                                          H, A, delta, I)
            observations[i] = x_update[0,0]
            hidden[i] = x_update[1,0]
        return (observations[1:], hidden[1:])
    
    # TODO: allow different optimizers
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
        return xopt
    
    def _time_update(self, params, x, F, H, P, U, Q):
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
        b0, b1, b2, b3, b4, b5 = self.bounds
        mu = periodic_map(params[0], b0[0], b0[1])
        kappa = periodic_map(params[1], b1[0], b1[1])
        theta = periodic_map(params[2], b2[0], b2[1])
        sigma = periodic_map(params[3], b3[0], b3[1])
        rho = periodic_map(params[4], b4[0], b4[1])
        v0 = periodic_map(params[5], b5[0], b5[1]) # ensure positive vt
        return mu, kappa, theta, sigma, rho, v0

class UKFHeston(object):
    def __init__(self, y, dt=1/250, is_log=False, bounds=None):
        self.y = y
        self.logS0 = np.log(y[0]) if not is_log else y[0]
        self.dt = dt

        # bounds for periodic map
        if bounds is not None:
            self.bounds = bounds 
        else:
            mu = (0.01, 1)
            kappa = (1, 3)
            theta = (1e-3, 0.2)
            sigma = (1e-3, 0.7)
            rho = (-1, 1)
            v0 = (1e-3, 0.2)
            self.bounds = [mu, kappa, theta, sigma, rho, v0]

    def obj(self, params):
        mu, kappa, theta, sigma, rho, v0 = self._unwrap_params(params)
        print(params)
        y_hat = self.logS0
        N = len(self.y)
        Ew, Ev, P, F, H = self._init_transitions(params)
        x_update = v0

        # init sigma points and weight updates params
        L = 2 # heston has two states
        K = 3-L
        alpha = 1e-3
        beta = 2
        x_sig, lda, W_m, W_c = self._init_weights(params, L, K, alpha, beta)
        obj = 0 # constant for making matrix Semi Pos. Def
        eps = 1e-6
        for i in range(1, N):
            # prediction
            dy = self.y[i] - self.y[i-1]
            F_x_sig, H_x_sig, x_pred, P_pred, y_hat = self._time_update(params, x_update, x_sig, W_m, W_c, F, H, self.y[i-1], dy, P, Ew, Ev, L, lda)

            # measurement update
            R = x_pred * self.dt
            x_update, P, Pyy = self._measurement_update(F_x_sig, H_x_sig, W_c, x_pred, P_pred, self.y[i], y_hat, R)
            
            # likelihood
            delta = self.y[i] - y_hat
            A = Pyy # don't need H to retrieve entries and R already added
            obj += np.log(np.fabs(A)) + delta**2/A
        return obj/N

    def optimize(self, init_params, maxiter=10000):
        """
        Performs simplex optimization for parameter estimation
        """
        mu, kappa, theta, sigma, rho, v0 = self._unwrap_params(init_params)
        # print([mu, kappa, theta, sigma, rho, v0])
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
        return xopt

    def filter(self, y, params):
        mu, kappa, theta, sigma, rho, v0 = self._unwrap_params(params)
        print("params: {}".format([mu, kappa, theta, sigma, rho, v0]))
        y_hat = self.logS0
        N = len(y)
        # initialize transitions
        Ew, Ev, P, F, H = self._init_transitions(params)

        # init state variables
        x_update = v0
        x_pred_vals = np.zeros(N)
        x_pred_vals[0] = x_update
        y_hat_vals = np.zeros(N) # predicted price
        y_hat_vals[0] = y_hat

        # init sigma points and weight updates params
        L = 2 # heston has two states
        K = 3-L
        alpha = 1e-3
        beta = 2
        x_sig, lda, W_m, W_c = self._init_weights(params, L, K, alpha, beta)

        eps = 1e-6 # constant for making matrix Semi Pos. Def
        for i in range(1, N):
            # prediction
            dy = y[i] - y[i-1]
            F_x_sig, H_x_sig, x_pred, P_pred, y_hat = self._time_update(params, x_update, x_sig, W_m, W_c, F, H, y[i-1], dy, P, Ew, Ev, L, lda)

            # measurement update
            R = x_pred * self.dt
            x_update, P, _ = self._measurement_update(F_x_sig, H_x_sig, W_c, x_pred, P_pred, y[i], y_hat, R)
            
            # append results
            x_pred_vals[i] = x_update
            y_hat_vals[i] = y_hat
        return x_pred_vals, y_hat_vals

    def _time_update(self, params, x_update, x_sig, W_m, W_c, F, H, y_prev, dy, P, Ew, Ev, L, lda):
        mu, kappa, theta, sigma, rho, v0 = self._unwrap_params(params)
        # init augmentation
        Q = sigma**2*(1-rho**2)*x_update*self.dt
        x_aug, P_aug = self._aug_state(params, x_update, Ew, P, Q)

        # make P positive definite
        # replace negative values in diagonal with small constant
        while not self._is_pos_def(P_aug):
            P_aug = P_aug + eps * np.eye(P_aug.shape[0])
        u_state = kappa*theta*self.dt - sigma*rho*mu*self.dt + sigma*rho*dy

        # generating sigma points
        # note P_aug is diagonal
        x_sig, rP_aug = self._generate_sigmas(x_aug, x_sig, P_aug, L, lda)

        # map sigma point through process transition and map points
        F_x_sig = self._transition(x_sig, u_state, F, Q)
        x_pred = max(1e-3, np.sum(W_m * F_x_sig))
        P_pred = np.sum(W_c * (F_x_sig - x_pred)**2) + Q

        # observations
        R = x_pred * self.dt
        x_pred_aug, P_pred_aug = self._aug_state(params, x_pred, Ev, P_pred, R)

        while not self._is_pos_def(P_pred_aug):
            P_pred_aug = P_pred_aug + eps * np.eye(P_pred_aug.shape[0])
        u_obs = y_prev + mu*self.dt
        
        x_sig, rP_pred_aug = self._generate_sigmas(x_pred_aug, x_sig, P_pred_aug, L, lda)
        H_x_sig = self._transition(x_sig, u_obs, H, R)
        y_hat = np.sum(W_m * H_x_sig)

        return F_x_sig, H_x_sig, x_pred, P_pred, y_hat

    def _measurement_update(self, F_sig, H_sig, W_c, x_pred, P_pred, y, y_hat, R):
        Pyy = np.sum(W_c*(H_sig-y_hat)**2) + R
        Pxy = np.sum(W_c*(F_sig-x_pred)*(H_sig-y_hat))
        K = Pxy/Pyy
        x_update = max(1e-3, x_pred + K*(y-y_hat))
        P = P_pred - K*Pyy*K
        return x_update, P, Pyy

    def _init_transitions(self, params):
        mu, kappa, theta, sigma, rho, v0 = self._unwrap_params(params)
        Ew = 0 # expectation of system (gaussian) noise is 0
        Ev = 0 # expectation of measurement (observables)
        P = 1e-5 # P0 covariance is 0
        F = 1 - kappa*self.dt + 1/2*rho*sigma*self.dt
        H = -1/2*self.dt
        return Ew, Ev, P, F, H

    def _init_weights(self, params, L, K, alpha=1e-3, beta=2):
        """
        Unscented weights initialization
        """
        lda = alpha**2*(L+K) - L
        x_sig = np.matrix(np.zeros((L,2*L+1)))
        W_m = np.zeros(2*L+1)
        W_c = np.zeros(2*L+1)
        W_m[0] = lda/(L+lda)
        W_c[0] = lda/(L+lda) + (1-alpha**2+beta)
        W_m[1:] = 1/(2*(L+lda))
        W_c[1:] = 1/(2*(L+lda))
        return x_sig, lda, W_m, W_c

    def _aug_state(self, params, x, expectation, cov_mat, noise_cov):
        mu, kappa, theta, sigma, rho, v0 = self._unwrap_params(params)
        x_aug = np.matrix([x, expectation]).T
        P_aug = np.matrix([[cov_mat, 0],
                           [0, noise_cov]])
        return x_aug, P_aug

    def _generate_sigmas(self, x, sigmas, aug_mat, L, lda):
        # generating sigma points
        # note P_aug is diagonal
        r_aug_mat = np.sqrt(L+lda) * np.sqrt(aug_mat)
        sigmas[:, 0] = x # set center
        for k in range(1, L+1):
            sigmas[:, k] = x + r_aug_mat[:, k-1]
            sigmas[:,L+k] = x - r_aug_mat[:, k-1]
        return sigmas, r_aug_mat

    def _transition(self, sigmas, u, transition_mat, noise_cov):
        """
        Transition sigma points with given transition matrix
        """
        mapped_sig = np.array(sigmas[0, :])[0]
        mapped_sig = transition_mat * mapped_sig + u + np.sqrt(noise_cov)*np.array(sigmas[1,:])[0]
        return mapped_sig

    def _is_pos_def(self, A):
        if np.array_equal(A, A.T):
            try:
                np.linalg.cholesky(A)
                return True
            except np.linalg.LinAlgError:
                return False
        return False

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
        b0, b1, b2, b3, b4, b5 = self.bounds
        mu = periodic_map(params[0], b0[0], b0[1])
        kappa = periodic_map(params[1], b1[0], b1[1])
        theta = periodic_map(params[2], b2[0], b2[1])
        sigma = periodic_map(params[3], b3[0], b3[1])
        rho = periodic_map(params[4], b4[0], b4[1])
        v0 = periodic_map(params[5], b5[0], b5[1])
        return mu, kappa, theta, sigma, rho, v0
    