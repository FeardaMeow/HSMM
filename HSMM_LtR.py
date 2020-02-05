import numpy as np

class HSMM_LtR():
    # Log Likelihood = - sum_t( log ( 1 / forward norm scaling ) )
    def __init__(self, N, A=None, pi=None):
        self.N = N
        self.A = A
        self.pi = pi

        # Function placeholders for user defined, follows scipy object methods
        self.f_obs = None
        self.obs_params = []

        self.f_duration = None
        self.duration_params = []
        
        # Time delta
        self.t_delta = None

        # Current likelihood
        self.likelihodd = None

    def train(self, x):
        # Initialize parameters

        # calculate observation probabilities

        # calculate forward probabilities

        # calculate backwards probabilities

        # calculate filtered probabilities

        # update observation parameters

        # update duration parameters

        # check convergence criteria

        pass
    
    def _initialize(self, x):
        # Initialize obs_params
        # Initialize duration_params
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

        # Calculate the dynamic transitsion matrix
        A = np.zeros((obs_probs.shape[0], self.N, self.N))
        A[0,:,:] = self._calculate_transition_matrix(duration_est[0,:])

        # Initialize the forward probabilities
        alpha = np.zeros(obs_probs.shape)
        alpha[0,:] = self.pi*obs_probs[0,:]

        log_likelihood = np.log(1/np.sum(alpha[0,:]))

        alpha[0,:] /= np.sum(alpha[0,:])

        for t in range(1,obs_probs.shape[0]):
            alpha[t,:] = alpha[t-1,:].dot(A[t-1,:,:])*obs_probs[t,:]
            alpha_sum = np.sum(alpha[t,:])
            log_likelihood += np.log(1/alpha_sum)
            alpha[t,:] /= alpha_sum
            # Estimate next duration
            duration_est[t,:] = self._estimate_duration(duration_est[t-1,:], A[t-1,:,:], alpha[t-1:t+1,:], obs_probs[t,:])
            A[t,:,:] = self._calculate_transition_matrix(duration_est[t,:])
        
        log_likelihood *= -1
        
        # Maximize log likelihood for best model
        return alpha, A, duration_est, log_likelihood

    def _backward(self, obs_probs, A):
        beta = np.ones(obs_probs.shape)
        for t in range(obs_probs.shape[0]-2, -1, -1):
            beta[t,:] = A[t,:,:].dot(obs_probs[t+1,:]*beta[t+1,:])

        return beta

    def _filtered(self, alpha, beta, obs_probs, A):
        '''
        Input:
            obs_probs: TxN matrix of the observation probabilities at each timestep for all hidden states
        Output:
            TxN array of the probabilities of being in state i at the 
        '''
        ##### Only needed to update the transition matrix #####
        # Multiply alpha across the rows of the transition matrix
        #prob_i_j = (alpha[:,:,np.newaxis]*A)[:-1,:,:] # T-1xNxN
        # Multiply obs_probs*beta across the columns of the transition matrix
        #prob_i_j = prob_i_j*(obs_probs*beta)[1:,np.newaxis,:] # T-1xNxN

        # Calculate the probability of being in state i at time t given all the observed data and model parameters
        prob_i = alpha*beta
        prob_i = prob_i / np.sum(prob_i, axis=1)[:,np.newaxis]

        return prob_i

    def _update_obs_params(self, x, prob_i):
        pass
    
    def _update_duration_params(self, alpha, beta, obs_probs, A, d):
        pass