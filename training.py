import HSMM_LtR as hsmm
from scipy.stats import norm
import numpy as np
import pickle as pk
import random
from sklearn.model_selection import train_test_split

class hsmm_model(hsmm.HSMM_LtR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _calc_obs_ss(self, x, prob_i):
        '''
        TODO: Ensure variance isnt smaller than 0.1
        '''
        denom_sum = np.sum(prob_i, axis=0)
        mean = np.sum(prob_i*x[:,np.newaxis], axis=0)/denom_sum
        variance = np.sum(prob_i*np.power(x[:,np.newaxis] - mean[np.newaxis,:],2), axis=0)/denom_sum

        mean[-1] = np.mean(x[-120:])
        variance[variance <= 0.1] = 0.1

        return mean, variance

    def _update_obs_params(self, obs_ss):
        # Unpack data
        obs_mean = [i[0] for i in obs_ss]
        obs_stdev = [np.sqrt(i[1]) for i in obs_ss]

        obs_mean = np.mean(np.array(obs_mean), axis=0)
        obs_stdev = np.mean(np.array(obs_stdev), axis=0)

        obs_params = [(i,j) for i,j in zip(obs_mean, obs_stdev)]
        self.obs_params = obs_params
    
    def _update_duration_params(self, duration_ss):
        # Unpack data
        duration_mean = [i[0] for i in duration_ss]
        duration_stdev = [np.sqrt(i[1]) for i in duration_ss]

        duration_mean = np.mean(np.array(duration_mean), axis=0)
        duration_stdev = np.mean(np.array(duration_stdev), axis=0)

        duration_params = [(i,j) for i,j in zip(duration_mean, duration_stdev)]
        self.duration_params = duration_params

def unpack_dict(x):
    '''
    Input:
        A dictionary with PID:array. The array is a list of tuples with (time series, take-over time).
    Output:
        A tuple(list of time series, list of take-over time targets)
    '''
    ts_list = []
    tot_list = []
    gap_list = []
    for key in x:
        for i in x[key]:
            ts_list.append(i[0])
            tot_list.append(i[1])
            gap_list.append(str(key) + '_' + str(round(i[2], 3)))

    return ts_list, tot_list, gap_list


def main():
    seed = 1234566
    np.random.seed(seed)
    random.seed(seed)

    '''
    obs_params = [(0,1), (4,1), (8,1)]
    duration_params = [(10,1), (15,1), (2,.1)]

    data, durations = hsmm.sim_data(500, norm, obs_params, norm, duration_params)

    model = hsmm_model(N=3, f_obs = norm, f_duration = norm, t_delta=1/60)
    model.fit(data, parallel=False)
    '''

    with open('final_data.pk', 'rb') as f:
        final_data = pk.load(f, encoding='latin1')

    x, y, stratify = unpack_dict(final_data)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=102, stratify=stratify)
    print(len(X_train), len(X_test))

    #print(train['PID01'][0], train['PID02'][0])
    #print(test['PID01'][0], test['PID02'][0])
    #with open('train.pk', 'wb') as f:
    #    pk.dump(train, f)

    #with open('test.pk', 'wb') as f:
    #    pk.dump(test, f)
    

if __name__ == "__main__":
    main()