import HSMM_LtR as hsmm
from scipy.stats import norm
import numpy as np

class hsmm_model(hsmm.HSMM_LtR):
    
    def __init__(self, N, A=None, pi=None, **kwargs):
        super().__init__(**kwargs)

    def _update_obs_params(self, x, prob_i):
        pass
    
    def _update_duration_params(self, alpha, beta, obs_probs, A, d):
        pass

    def _initialize(self, x):
        # Initialize obs_params
        # Initialize duration_params
        pass

def sim_data(num_ts, state_density, state_params, duration_density, duration_params):
    data = []
    durations = []
    for i in range(num_ts):
        duration_list = [int(np.abs(duration_density.rvs(*i)*60)) for i in duration_params]
        data_temp = [state_density.rvs(*i, size = d) for d,i in zip(duration_list, state_params)]
        data.append(np.concatenate(data_temp))
        durations.append(duration_list)

    return data, durations

def main():
    np.random.seed(1234566)
