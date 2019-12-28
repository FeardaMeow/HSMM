import numpy as np

class HSMM_LtR():

    def __init__(self, N, A=None, pi=None):
        self.N = N
        self.A = A
        self.pi = pi

        # Function placeholders for user defined, follows scipy object methods
        self.f_obs = None
        self.obs_params = None

        self.duration_params = None
        self.f_duration = None

        # Time delta
        self.t_delta = None

    def train(self):
        pass

    def _calculate_transition_matrix(self, d):
        '''
        Input:
            d: Nx1 array of the current timestep duration estimates
        Output:
            N x N matrix of the current timesteps dynamic transition probabilities
        '''
        duration_probability = [(1 - self.f_duration.cdf(d[i], *self.duration_params[i]))/(1 - self.f_duration.cdf(d[i] - self.t_delta, *self.duration_params[i])) for i in range(self.N)]
        self_transition_matrix = np.eye(self.N) * duration_probability 
        return self_transition_matrix + (np.eye(self.N) - self_transition_matrix).dot(self.A)

    def _estimate_duration(self, d, A, alpha, obs_probs):
        '''
        Input:
            d: Nx1 array of the previous timestep duration estimates
            A: NxN matrix of the previous timestep dynamic transition matrix
            alpha: 2xN matrix of the previous and current timestep forward probabilities
            obs_probs: Nx1 array of the current timestep observation probabilities
        Output:
            N x 1 array for the estimated duration for the current timestep
        '''
        d_est = ((A.dot(alpha[0,:]).dot(obs_probs))/alpha[1,:])*(d+self.t_delta)
        return d_est

    def _forward(self):
        pass

    def _backward(self):
        pass

