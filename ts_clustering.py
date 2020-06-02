from tslearn.metrics import dtw
from sklearn.cluster import DBSCAN
from tslearn.barycenters import dtw_barycenter_averaging
import numpy as np

def find_indices(stratify):
    '''
    Input:
        List of pid labels as "PIDXX_[ACC GAP]"
    Output:
        Dictionary of PIDXX:[list of indices]
    '''
    pid_count = {}

    for i,j in enumerate(stratify):
        pid = j.split('_')[0]
        if pid not in pid_count:
            pid_count[pid] = [i]
        else:
            pid_count[pid].append(i)
            
    return pid_count

def ts_average(x, pid_indices):
    '''
    Input:
        x: [[time series 1], ... , [time series n]]
        pid_indices: {pid:[indicies]}
    Output:
        Average time series for each unique PID in PID_indices
        The pid label for each of the average time series
    '''
    dtw_avg = []
    dtw_avg_pid = []

    for i in pid_indices:
        dtw_avg.append(dtw_barycenter_averaging([x[j] for j in pid_indices[i]]))
        dtw_avg_pid.append(i)
        
    return dtw_avg, dtw_avg_pid

def create_clusters(x, stratify, **kwargs):
    # Find indices for each PID in dataset
    pid_indices = find_indices(stratify)
    # Create DTW Average
    pid_averages, pid_labels = ts_average(x, pid_indices)
    # Create distance matrix with DTW on averages
    dtw_matrix = np.zeros((len(pid_averages),len(pid_averages)))
    for i in range(len(pid_averages)):
        for j in range(len(pid_averages)):
            dtw_matrix[i,j] = dtw(pid_averages[i], pid_averages[j])
    # Cluster using DBSCAN
    clusters = DBSCAN(**kwargs).fit(dtw_matrix)
    # Create dictionary of clusters and pid labels
    label_dict = {}
    for i,j in zip(clusters.labels_, pid_labels):
        label_dict[j] = i
    # Pass back indices for the original data based on clusters
    temp_labels = np.array([i.split('_')[0] for i in stratify])
    
    final_dict = {}
    for i in label_dict:
        if label_dict[i] not in final_dict:
            final_dict[label_dict[i]] = np.where(temp_labels == i)[0]
        else:
            final_dict[label_dict[i]] = np.concatenate((final_dict[label_dict[i]], np.where(temp_labels == i)[0]))
            
    return final_dict