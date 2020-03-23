import HSMM_LtR as hsmm
from scipy.stats import norm
import numpy as np
import pickle as pk
import random

class hsmm_model(hsmm.HSMM_LtR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _calc_obs_ss(self, x, prob_i):
        denom_sum = np.sum(prob_i, axis=0)
        mean = np.sum(prob_i*x[:,np.newaxis], axis=0)/denom_sum
        variance = np.sum(prob_i*np.power(x[:,np.newaxis] - mean[np.newaxis,:],2), axis=0)/denom_sum

        return mean, variance

    def _update_obs_params(self, obs_ss):
        pass
    
    def _update_duration_params(self, duration_ss):
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

def train_test_split(data, p=3):
    '''
    Input:
        A dictionary with PID:array. The array is a list of tuples with (time series, take-over time).
    Output:
        (train dictionary, test dictionary)
    '''
    train_dict = {}
    test_dict = {}
    for key in data:
        temp_list = data[key]
        random.shuffle(temp_list)
        train_dict[key] = temp_list[:-p]
        test_dict[key] = temp_list[-p:]

    return train_dict, test_dict

def unpack_dict(x):
    '''
    Input:
        A dictionary with PID:array. The array is a list of tuples with (time series, take-over time).
    Output:
        A tuple(list of time series, list of take-over time targets)
    '''
    ts_list = []
    tot_list = []
    for key in x:
        for i in x[key]:
            ts_list.append(i[0])
            tot_list.append(i[1])

    return ts_list, tot_list

def main():
    seed = 1234566
    np.random.seed(seed)
    random.seed(seed)
    with open('final_data.pk', 'rb') as f:
        final_data = pk.load(f, encoding='latin1')

    # Train Test Split
    train, test = train_test_split(final_data, p=3)

    print(train['PID01'][0], train['PID02'][0])
    print(test['PID01'][0], test['PID02'][0])
    with open('train.pk', 'wb') as f:
        pk.dump(train, f)

    with open('test.pk', 'wb') as f:
        pk.dump(test, f)
    

if __name__ == "__main__":
    main()