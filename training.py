import HSMM_LtR as hsmm 

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