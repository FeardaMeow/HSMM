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

    def train(self):
        pass

    def _calculate_transition_matrix(self):
        pass

    def _estimate_duration(self):
        pass

    def _forward(self):
        pass

    def _backward(self):
        pass

    