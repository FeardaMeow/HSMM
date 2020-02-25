import HSMM_LtR as hsmm
from scipy.stats import norm, halfnorm
import numpy as np
import pytest

# returns and instance of the HSMM_LtR class and observation
def pytest_funcarg__test_data(request):
    hsmm_model = hsmm.HSMM_LtR(N=3, f_obs = norm, obs_params = [(),(),()], f_duration = halfnorm, duration_params = [(),(),()])
    return 

def test_transition_matrix(self):
    pass

def test_estimate_duration(self):
    pass

def test_forward(self):
    pass

def test_backward(self):
    pass

def test_filtered(self):
    pass

def test_initialize(self):
    pass

