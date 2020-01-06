import numpy as np

class HSMM_LtR():
    # Log Likelihood = - sum_t( log ( 1 / forward norm scaling ) )
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
        duration_probability = [(1 - self.f_duration.cdf(d[i], *self.duration_params[i]))/np.max(1 - self.f_duration.cdf(d[i] - self.t_delta, *self.duration_params[i]),0.0001) for i in range(self.N)]
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
        d_est = (np.diag(A)*alpha[0,:]*obs_probs/alpha[1,:])*(d + self.t_delta)
        return d_est

    def _forward(self, obs_probs):
        '''
        Input:
            obs_probs: TxN matrix of the observation probabilities at each timestep for all hidden states
        Output:
            tuple of 4 values 
            (forward probabilities matrix (TxN), 
            dyanmic transition matrices (NxNxT), 
            duration estimate matrix (TxN),
            the log likelihood of the model)
        '''
        # Initialize the duration estimates
        duration_est = np.ones(obs_probs.shape)*self.t_delta

        # Initialize the dynamic transition matrix
        A_dt = np.zeros((self.N, self.N, obs_probs.shape[0]))
        A_dt[:,:,0] = self._calculate_transition_matrix(duration_est[0,:])

        # Initialize the forward probabilities
        alpha = np.zeros(obs_probs.shape)
        alpha[0,:] = self.pi*obs_probs[0,:]

        log_likelihood = np.log(1/np.sum(alpha[0,:]))

        alpha[0,:] /= np.sum(alpha[0,:])
        # Log Likelihood = - sum_t( log ( 1 / forward norm scaling ) )

        for t in range(1,obs_probs.shape[0]):
            alpha[t,:] = alpha[t-1,:].dot(A_dt[:,:,t-1])*obs_probs[t,:]
            log_likelihood += np.log(1/np.sum(alpha[t,:]))
            alpha[t,:] /= np.sum(alpha[t,:])
            # Estimate next duration
            duration_est[t,:] = self._estimate_duration(duration_est[t-1,:], A_dt[:,:,t-1], alpha[t-1:t+1,:], obs_probs[t,:])
            A_dt[:,:,t] = self._calculate_transition_matrix(duration_est[t,:])
        
        log_likelihood *= -1
        
        return(alpha, A_dt, duration_est, log_likelihood)

    def _backward(self, obs_probs, A_dt):
        beta = np.ones(obs_probs.shape)
        for t in range(obs_probs.shape[0]-2, -1, -1):
            beta[t,:] = A_dt[:,:,t].dot(obs_probs[t+1,:]*beta[t+1,:])

        return beta
