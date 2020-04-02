import numpy as np
from typing import TypeVar, List, Tuple
from nptyping import Array
from random import sample
from scipy.stats import norm
from tqdm import tqdm
from numba import jit
import multiprocessing as mp

class HSMM_LtR():
    # Log Likelihood = - sum_t( log ( 1 / forward norm scaling ) )
    def __init__(self, N: int, f_obs, f_duration, obs_params = None, duration_params = None, t_delta:float=1/60, A=None, pi=None) -> None:
        self.N = N
        self.A = A
        self.pi = pi

        # Function placeholders for user defined, follows scipy object methods
        self.f_obs = f_obs
        self.obs_params = obs_params

        self.f_duration = f_duration
        self.duration_params = duration_params
        
        # Time delta
        self.t_delta = t_delta

        # Current likelihood
        self.likelihood = None

        # Time after TOT in seconds
        self.T_after = 120

    def _calc_probs(self, x, f='observation'):
        probs_list = []
        if f == 'observation':
            for params in self.obs_params:
                probs_list.append(self.f_obs.pdf(x, *params))
        
        return np.vstack(probs_list).T

    def fit(self, x: List[Array[float]], n:int=100, parallel:bool=False):
        # Initialize parameters
        self._initialize(x)
        counter = 0

        while counter < n:
            overall_likelihood = 0
            np.random.shuffle(x)
            duration_ss = []
            obs_ss = []
            if parallel:
                pass
            else:
                for x_i in tqdm(x):
                    # calculate observation probabilities
                    obs_probs = self._calc_probs(x_i, 'observation')

                    # calculate forward probabilities
                    forward_probs, A, duration_est, log_likelihood = self._forward(obs_probs)
                    overall_likelihood += log_likelihood

                    # calculate backwards probabilities
                    backward_probs = self._backward(obs_probs, A)

                    # calculate filtered probabilities
                    prob_i = self._filtered(forward_probs, backward_probs, obs_probs, A)

                    # calculate the duration sufficient statistics
                    duration_ss.append(self._calc_duration_ss(forward_probs, backward_probs, obs_probs, duration_est, A))

                    # calculate the obs sufficient statistics
                    obs_ss.append(self._calc_obs_ss(x_i, prob_i))
            
            print('Iteration ' + str(counter+1) + ': ' + str(overall_likelihood))
            #print("Elapsed = %s" % (end - start))
            
            # check convergence criteria
            if self.likelihood != None and overall_likelihood < self.likelihood:
                break
            
            self.likelihood = overall_likelihood

            # update observation parameters
            self._update_obs_params(obs_ss)

            # update duration parameters
            self._update_duration_params(duration_ss)

            print('Observation Parameters:')
            print(self.obs_params)
            print('Duration Parameters:')
            print(self.duration_params)

            counter += 1
    
    @jit()
    def _fit_parallel(self, x: List[Array[float]]):
        duration_ss = []
        obs_ss = []
        overall_likelihood = 0
        for x_i in x:
            # calculate observation probabilities
            obs_probs = self._calc_probs(x_i, 'observation')

            # calculate forward probabilities
            forward_probs, A, duration_est, log_likelihood = self._forward(obs_probs)
            overall_likelihood += log_likelihood

            # calculate backwards probabilities
            backward_probs = self._backward(obs_probs, A)

            # calculate filtered probabilities
            prob_i = self._filtered(forward_probs, backward_probs, obs_probs, A)

            # calculate the duration sufficient statistics
            duration_ss.append(self._calc_duration_ss(forward_probs, backward_probs, obs_probs, duration_est, A))

            # calculate the obs sufficient statistics
            obs_ss.append(self._calc_obs_ss(x_i, prob_i))

        return overall_likelihood, duration_ss, obs_ss

    def _calc_duration_ss(self, forward_probs, backward_probs, obs_probs, duration_est, A) -> Tuple[Array[float],Array[float]]:
        '''
        TODO: Ensure variance isnt smaller than 0.1
        '''
        numerator = forward_probs[:-1,:]*np.einsum('tij,ti->ti',(np.ones((self.N,self.N)) - np.eye(self.N))[np.newaxis,:,:]*A[:-1,:,:],obs_probs[1:,:]*backward_probs[1:,:])
        num_sum = np.sum(numerator, axis=0)
        num_sum[num_sum == 0] = 1

        expected_value = np.sum(numerator*duration_est[:-1,:], axis=0)
        expected_value = expected_value/num_sum

        variance = np.sum(numerator*np.power(duration_est[:-1,:] - expected_value[np.newaxis,:],2), axis=0)
        variance = variance/num_sum

        # Set the last state values
        expected_value[-1] = self.T_after*(1/60)
        variance[-1] = 0.5

        variance[variance <= 1] = 1

        return expected_value, variance
    
    def _initialize(self, x:List[Array[float]], n:float = 0.1) -> None:
        # Initialize A
        if self.A is None:
            self.A = np.zeros((self.N, self.N))
            rng = np.arange(self.N-1)
            self.A[rng, rng+1] = 1
            self.A[-1,-1] = 1

        # Initialize pi
        if self.pi is None:
            self.pi = np.zeros(self.N)
            self.pi[0] = 1

        
        # Initialize obs_params and duration params
        N = int(n*len(x))
        init_list = sample(x, N)
        obs_params = []
        duration_params = []

        for n_array in init_list:
            # np.average with weights the length of the array
            split_arrays = np.array_split(n_array[:-self.T_after], self.N-1)
            obs_temp = []
            duration_temp = []
            for i in split_arrays:
                obs_temp.append(np.mean(i))
                duration_temp.append(i.shape[0]/60.0)
            obs_temp.append(np.mean(n_array[-self.T_after:]))
            duration_temp.append(self.T_after/60.0)

            # Store values
            obs_params.append(obs_temp)
            duration_params.append(duration_temp)

        # Calculate starting parameters and set them
        obs_params = np.asarray(obs_params)
        duration_params = np.asarray(duration_params)

        obs_mean = np.mean(obs_params, axis=0)
        obs_variance = np.std(obs_params, axis=0)
        
        duration_mean = np.average(duration_params, axis=0)
        duration_variance = np.std(duration_params, axis=0)
        duration_variance[-1] = 0.1

        self.obs_params = [(i,j) for i,j in zip(obs_mean, obs_variance)]
        self.duration_params = [(i,j) for i,j in zip(duration_mean, duration_variance)]

    def _calculate_transition_matrix(self, d):
        '''
        Input:
            d: Nx1 array of the current timestep duration estimates
        Output:
            N x N matrix of the current timesteps dynamic transition probabilities
        '''
        duration_probability = []
        for i in range(int(self.N)):
            numerator = 1 - self.f_duration.cdf(d[i], *self.duration_params[i])
            denom = max(1 - self.f_duration.cdf(d[i] - self.t_delta, *self.duration_params[i]), 0.001)
            duration_probability.append(numerator/denom)
        self_transition_matrix = np.eye(self.N) * np.array(duration_probability)
        self_transition_matrix = self_transition_matrix + np.dot(np.eye(self.N) - self_transition_matrix, self.A)
        # Normalize the rows of the transition matrix to sum to 1
        return self_transition_matrix/np.sum(self_transition_matrix, axis=1)[:,np.newaxis]

    def _estimate_duration(self, d: Array[float], A: Array[float], alpha, obs_probs) -> Array[float]:
        '''
        Input:
            d: Nx1 array of the previous timestep duration estimates
            A: NxN matrix of the previous timestep dynamic transition matrix
            alpha: 2xN matrix of the previous and current timestep forward probabilities
            obs_probs: Nx1 array of the current timestep observation probabilities
        Output:
            N x 1 array for the estimated duration for the current timestep
        '''
        #d_est = ((np.diag(A)*alpha[0,:]*obs_probs)/alpha[1,:])*d + self.t_delta
        # Azimi et al.
        d_est = alpha[0,:]*d + self.t_delta
        return d_est

    def _forward(self, obs_probs: Array[float]) -> Tuple[Array[float], Array[float], Array[float], float]:
        '''
        Input:
            obs_probs: TxN matrix of the observation probabilities at each timestep for all hidden states
        Output:
            tuple of 4 values 
            (forward probabilities matrix (TxN), 
            dyanmic transition matrices (NxNxT), 
            duration estimate matrix (TxN),
            the log likelihood of the model)S
        '''
        # Initialize the duration estimates
        duration_est = np.ones(obs_probs.shape)*self.t_delta

        # Calculate the dynamic transitsion matrix
        A = np.zeros((obs_probs.shape[0], self.N, self.N))
        A[0,:,:] = self._calculate_transition_matrix(duration_est[0,:])

        # Initialize the forward probabilities
        alpha = np.zeros(obs_probs.shape)
        alpha[0,:] = self.pi*obs_probs[0,:]

        alpha_sum = np.sum(alpha[0,:])

        if 1/alpha_sum == 1:
            alpha_sum += 0.001

        log_likelihood = np.log(1/alpha_sum)

        alpha[0,:] /= alpha_sum

        for t in range(1,obs_probs.shape[0]):
            alpha[t,:] = np.dot(alpha[t-1,:],A[t-1,:,:])*obs_probs[t,:]
            alpha_sum = np.sum(alpha[t,:])

            alpha_sum += 0.00001

            if 1/alpha_sum == 1:
                alpha_sum += 0.001

            log_likelihood += np.log(1/alpha_sum)
            # Check if any alpha is zero and inject a small probability = 0.001
            alpha[t,:][np.isnan(alpha[t,:])] = 0
            alpha[t,:] += np.max(alpha[t,:])/100
            alpha[t,:] /= np.sum(alpha[t,:])
            # Estimate next duration
            duration_est[t,:] = self._estimate_duration(duration_est[t-1,:], A[t-1,:,:], alpha[t-1:t+1,:], obs_probs[t,:])
            A[t,:,:] = self._calculate_transition_matrix(duration_est[t,:])
        
        log_likelihood *= -1
        
        # Maximize log likelihood for best model
        return alpha, A, duration_est, log_likelihood

    @staticmethod
    @jit(nopython=True)
    def _backward(obs_probs: Array[float], A: Array[float]):
        beta = np.ones(obs_probs.shape)
        for t in range(obs_probs.shape[0]-2, -1, -1):
            beta[t,:] = np.dot(A[t,:,:], obs_probs[t+1,:]*beta[t+1,:])
            beta[t,:][np.isnan(beta[t,:])] = 0
            beta[t,:] += 0.001
            beta[t,:] /= np.sum(beta[t,:])

        return beta

    def _filtered(self, forward_probs, backward_probs, obs_probs, A: Array[float]):
        '''
        Input:
            alpha:
            beta:
            obs_probs: TxN matrix of the observation probabilities at each timestep for all hidden states
            A:
        Output:
            TxN array of the probabilities of being in state i at the 
        '''
        ##### Only needed to update the transition matrix #####
        # Multiply forward_probs across the rows of the transition matrix
        #prob_i_j = (forward_probs[:,:,np.newaxis]*A)[:-1,:,:] # T-1xNxN
        # Multiply obs_probs*backward_probs across the columns of the transition matrix
        #prob_i_j = prob_i_j*(obs_probs*backward_probs)[1:,np.newaxis,:] # T-1xNxN

        # Calculate the probability of being in state i at time t given all the observed data and model parameters
        prob_i = forward_probs*backward_probs
        prob_i = prob_i / np.sum(prob_i, axis=1)[:,np.newaxis]

        return prob_i

    def _viterbi(self, x):
        # observation probabilities
        obs_probs = self._calc_probs(x, 'observation')

        # calculate forward probabilities
        _, A, duration_est, _ = self._forward(obs_probs)

        T_1 = np.zeros((x.shape[0], self.N))
        T_2 = np.zeros((x.shape[0], self.N))

        T_1[0,:] = self.pi * obs_probs[0,:]

        for t in range(1,x.shape[0]):
            T_1[t,:] = np.max(T_1[t-1,:][:,np.newaxis] * (A[t-1,:,:] * obs_probs[t,:][np.newaxis,:]), axis=0)
            T_1[t,:] += 0.001
            T_1[t,:] = T_1[t,:]/np.sum(T_1[t,:])
            T_2[t,:] = np.argmax(T_1[t-1,:][:,np.newaxis] * (A[t-1,:,:] * obs_probs[t,:][np.newaxis,:]), axis=0)
        
        #best_path_prob = np.max(T_1[-1,:])
        best_path_pointer = np.argmax(T_1[-1,:])

        return T_1[-1,:], best_path_pointer, duration_est[-1,:]

    def predict(self, x, ci=0.95):
        ci_std_value = round(norm.ppf(ci + (1-ci)/2),2)

        d_avg = 0
        d_low = 0
        d_high = 0

        # Unpack duration params
        duration_mean = [i[0] for i in self.duration_params]
        duration_stdev = [i[1] for i in self.duration_params]

        duration_mean = np.array(duration_mean[:-1])
        duration_stdev = np.array(duration_stdev[:-1])

        # Initialize counter and current state tracker
        n = 0
        current_prob, current_state, current_duration = self._viterbi(x)

        while n <= self.N and current_state != self.N-1:
            if n == 1:
                d_avg = d_avg + np.sum((duration_mean - current_duration[:-1])*current_prob[:-1])
                d_low = d_low + np.sum(np.clip((duration_mean - duration_stdev*ci_std_value - current_duration[:-1])*current_prob[:-1]), 0, None)
                d_high = d_high + np.sum((duration_mean + duration_stdev*ci_std_value - current_duration[:-1])*current_prob[:-1])
            else:
                d_avg = d_avg + np.sum((duration_mean)*current_prob[:-1])
                d_low = d_low + np.sum(np.clip((duration_mean - duration_stdev*ci_std_value)*current_prob[:-1]), 0, None)
                d_high = d_high + np.sum((duration_mean + duration_stdev*ci_std_value)*current_prob[:-1])

            current_prob = np.dot(self.A.T, current_prob)
            current_state = np.argmax(current_prob, axis=0)

            n+=1

        return d_avg, d_low, d_high

    def _calc_obs_ss(self, x, prob_i):
        '''
        Input:
            x: 1 obervation time series
            prob_i: the filtered probabilities of being in state i and time t
        Output:
            Tuple(expected value, variance)
        '''
        pass

    def _update_obs_params(self, obs_ss):
        '''
        Input:
            obs_ss: list of tuples(expected value, variance)
        Output:
            None: will update obs_params
        '''
        pass
    
    def _update_duration_params(self, duration_ss):
        '''
        Input:
            duration_ss: list of tuples(expected value, variance)
        Output:
            None: Will update self.duration_params
        '''
        pass

def main():
    from scipy.stats import norm
    from matplotlib import pyplot as plt

    np.random.seed(123)

    data_index = 123
    obs_params = [(0,1), (4,1), (8,1)]
    duration_params = [(10,2), (15,2), (2,.1)]

    data, durations = sim_data(500, norm, obs_params, norm, duration_params)

    hsmm = HSMM_LtR(N=3, f_obs = norm, f_duration = norm)
    hsmm._initialize(data, n=0.5)

    obs_probs = hsmm._calc_probs(data[data_index])
    alpha, A, duration_est, log_likelihood = hsmm._forward(obs_probs)

    print(log_likelihood)

    hsmm.duration_params = duration_params
    hsmm.obs_params = obs_params
    '''
    hsmm.A = np.array([[0,1,0],[0,0,1],[0,0,1]])
    hsmm.pi = np.array([1,0,0])
    '''

    obs_probs = hsmm._calc_probs(data[data_index])
    alpha, A, duration_est, log_likelihood = hsmm._forward(obs_probs)
    beta = hsmm._backward(obs_probs, A)

    print(log_likelihood)

    d_mean, d_var = hsmm._calc_duration_ss(alpha, beta, obs_probs, duration_est, A)

    print(d_mean, d_var)

    v_prob, v_state, _ = hsmm._viterbi(data[data_index][0:20])

    print(v_prob, v_state)
    

def sim_data(num_ts, state_density, state_params, duration_density, duration_params, realistic=True):
    data = []
    durations = []
    for _ in range(num_ts):
        duration_list = [int(np.abs(duration_density.rvs(*i)*60)) for i in duration_params]
        if realistic:
            duration_list[-1] = 120
        data_temp = [state_density.rvs(*i, size = d) for d,i in zip(duration_list, state_params)]

        data.append(np.concatenate(data_temp))
        durations.append(np.array(duration_list))

    return data, durations

if __name__ == "__main__":
    main()