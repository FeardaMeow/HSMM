import HSMM_LtR as hsmm
from scipy.stats import norm
import numpy as np
import pickle as pk
import random
from sklearn.model_selection import StratifiedShuffleSplit
from operator import itemgetter 
import pandas as pd
import os

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

        mean[-1] = np.mean(x[-self.T_after:])
        variance[variance <= 0.0025] = 0.0025

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

        duration_mean = [i for i in duration_ss]
        #duration_stdev = [np.sqrt(i[1]) for i in duration_ss]

        duration_stdev = np.std(np.array(duration_mean), axis=0)
        duration_mean = np.mean(np.array(duration_mean), axis=0)
        duration_stdev[-1] = 1

        if np.sum(duration_mean)-1 < 6:
            duration_params = [(i,j) for i,j in zip(duration_mean, duration_stdev)]

            self.duration_params = duration_params

    def _update_online_params(self, obs_ss, duration_ss, learning_rate):
        '''
        Input:
            duration_ss: list of tuples(expected value, variance)
            obs_ss: list of tuples(expected value, variance)
            learning_rate: float
        Output:
            None: Will update self.duration_params
        '''
        # Obs parameters
        obs_mean = [i[0] for i in obs_ss]
        obs_stdev = [np.sqrt(i[1]) for i in obs_ss]

        obs_mean = np.mean(np.array(obs_mean), axis=0)
        obs_stdev = np.mean(np.array(obs_stdev), axis=0)

        # Unpack current obs params
        current_obs_mean = np.array([i[0] for i in self.obs_params])
        current_obs_stdev = np.array([i[1] for i in self.obs_params])

        # Duration parameters
        duration_mean = [i for i in duration_ss]

        duration_stdev = np.std(np.array(duration_mean), axis=0)
        duration_mean = np.mean(np.array(duration_mean), axis=0)
        duration_stdev[-1] = 1

        # Unpack current duration params
        current_duration_mean = np.array([i[0] for i in self.duration_params])
        current_duration_stdev = np.array([i[1] for i in self.duration_params])

        # Update
        new_duration_mean = current_duration_mean*(1-learning_rate) + learning_rate*duration_mean
        new_duration_stdev = current_duration_stdev*(1-learning_rate) + learning_rate*duration_stdev

        new_obs_mean = current_obs_mean*(1-learning_rate) + learning_rate*obs_mean
        new_obs_stdev = current_obs_stdev*(1-learning_rate) + learning_rate*obs_stdev

        duration_params = [(i,j) for i,j in zip(new_duration_mean, new_duration_stdev)]

        obs_params = [(i,j) for i,j in zip(new_obs_mean, new_obs_stdev)]

        self.duration_params = duration_params
        self.obs_params = obs_params

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
    split_list = []
    for key in x:
        for i in x[key]:
            ts_list.append(i[0])
            tot_list.append(i[1])
            gap_list.append(str(key) + '_' + str(round(i[2], 3)))
            split_list.append(i[3])

    return ts_list, tot_list, gap_list, split_list

def main():
    seed = 1234566
    np.random.seed(seed)
    random.seed(seed)

    pid_range = ['PID01','PID02','PID03','PID04','PID05',
                'PID06','PID07','PID08','PID09','PID10',
                'PID11','PID12','PID13','PID14','PID15',
                'PID16','PID17','PID18','PID19','PID20',
                'PID21','PID22','PID23','PID24','PID25',
                'PID26','PID27','PID28','PID29','PID30',
                'PID31','PID32','PID33','PID34']
   
    '''
    with open('final_data_ds.pk', 'rb') as f:
        final_data = pk.load(f, encoding='latin1')

    x, y, stratify, splits = unpack_dict(final_data)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=102)
    for i,j in sss.split(np.zeros(len(stratify)), stratify):
        train = i
        test = j

    # Train Test Split

    X_train = list(itemgetter(*train)(x)) 
    X_test = list(itemgetter(*test)(x)) 
    y_train = list(itemgetter(*train)(y)) 
    y_test = list(itemgetter(*test)(y)) 
    stratify_train = list(itemgetter(*train)(stratify)) 
    stratify_test = list(itemgetter(*test)(stratify)) 
    
    
    with open('X_train_rate.pk', 'wb') as f:
        pk.dump(X_train, f)

    with open('y_train_rate.pk', 'wb') as f:
        pk.dump(y_train, f)

    with open('X_test_rate.pk', 'wb') as f:
        pk.dump(X_test, f)

    with open('y_test_rate.pk', 'wb') as f:
        pk.dump(y_test, f)

    with open('stratify_train_rate.pk', 'wb') as f:
        pk.dump(stratify_train, f)

    with open('stratify_test_rate.pk', 'wb') as f:
        pk.dump(stratify_test, f)
    
    with open('X_train.pk', 'rb') as f:
        X_train = pk.load(f)
    '''

    with open('X_train_rate.pk', 'rb') as f:
        X_train = pk.load(f)
    
    with open('stratify_train_rate.pk', 'rb') as f:
        stratify_train = pk.load(f)
    
    for i in range(len(X_train)):
        X_train[i] = X_train[i][:-20]

    pid_list = []
    gap_list = []
    for i in stratify_train:
        pid_list.append(i.split('_')[0])
        gap_list.append(i.split('_')[1])

    with open('models/offline/model_train_N5_nc_1.pk', 'rb') as f:
        starting_model = pk.load(f)
    # rate (0,1), alpha (0.5,1]
    # 0.30
    model_params = {'learning_rate':0.4, 
                    'learning_rate_alpha':0.5, 
                    'm':9, 
                    'm_overlap':2}

    for i in pid_range:
        # Initial starting model
        model = starting_model

        # Partition data
        X_online = [X_train[j] for j,d in enumerate(pid_list) if d == i]

        model.fit(X_online, online=True, **model_params)
        print("Final")
        print("Observation Parameters:")
        print(model.obs_params)
        print("Duration Parameters:")
        print(model.duration_params)

        with open(os.path.join('models', 'online' , i + '_1.pk'), 'wb') as f:
            pk.dump(model, f)
    
if __name__ == "__main__":
    main()
