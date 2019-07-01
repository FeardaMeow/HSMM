import scipy.stats as st
from sklearn.mixture import GaussianMixture #GMM
import numpy as np #Utility
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import copy
import pickle

###### FUNCTIONS + CLASSSES START #######
def segmentData(label):
    temp = []
    last_label = label[0]
    for index, i in enumerate(label):
        if i != last_label:
            last_label = i
            temp.append(index)
    return(np.array(temp))

def isPSD(A, tol=1e-8):
    E,V = st.linalg.eigh(A)
    return np.all(E > -tol)
#d is a vector state duration times for each state 
#gmmhsmm is a object of class GMM_HSMM   

#Need to fix so it is non-increasing
def dynTranMatrix(d, gmmhsmm): 
    Pd = [(1.001-st.norm.cdf(d[i], loc=gmmhsmm.sd_mean[0,i], scale=gmmhsmm.sd_var[0,i]))/(1.001-st.norm.cdf(d[i]-1.0/59.0, loc=gmmhsmm.sd_mean[0,i], scale=gmmhsmm.sd_var[0,i])) for i in range(gmmhsmm.N)]
    Pd = np.diag(Pd)
    return(Pd + np.matmul(np.identity(gmmhsmm.N) - Pd,gmmhsmm.A)) 


def forward(gmmhsmm, b, d):
    alpha = np.zeros((b.shape[0],  gmmhsmm.N))
    A = np.zeros((b.shape[0],  gmmhsmm.N, gmmhsmm.N))
    A[0,:,:] = dynTranMatrix(d, gmmhsmm)
    dt = np.zeros((b.shape[0],  gmmhsmm.N))
    dt[0,:]=d
    
    alpha_lik = np.zeros((2, gmmhsmm.N))
    
    lik = 0
    print()
    for t in range(b.shape[0]):
        if t==0:
            alpha[t,:] =  gmmhsmm.pi*b[t,:]
            alpha[t,alpha[t,:]==0]=1
            alpha[t,:] /= np.sum(alpha[t,:])
        else:
            alpha[t,:] = np.dot(alpha[t-1,:],A[t-1,:,:])*b[t,:]
            alpha[t,alpha[t,:]==0]=1
            
            dt[t,:] = ((np.diag(A[t-1,:,:])*alpha[t-1,:]*b[t,:])/alpha[t,:])*(dt[t-1,:]+1/59)
            
            if t >= b.shape[0]-100:
                if t==b.shape[0]-1:
                    lik = np.log(np.sum(alpha_lik[0,:]))
                elif t == b.shape[0]-100:
                    alpha_lik[0,:] = alpha[t,:] 
                else:
                    alpha_lik[1,:] = np.dot(alpha_lik[0,:],A[t-1,:,:])*b[t,:]
                    alpha_lik[0,:] = alpha_lik[1,:]
            
            alpha[t,:] /= np.sum(alpha[t,:])
            A[t,:,:] = dynTranMatrix(dt[t,:], gmmhsmm)
    
    return(alpha, A, dt, lik)


def backward(gmmhsmm, b, A):
    beta = np.ones((b.shape[0],  gmmhsmm.N))
    for t in range(b.shape[0]-2,-1,-1):
        beta[t,:] = np.sum(A[t,:,:].T*(beta[t+1,:]*b[t+1,:]), axis=0)
        beta[t,beta[t,:]==0] = 1
        beta[t,:] /= np.sum(beta[t,:])
    return(beta)


def likelihood(gmmhsmm, x):
    b = np.zeros((np.atleast_2d(x).shape[0], gmmhsmm.N))
    
    for n in range(gmmhsmm.N):
        for m in range(gmmhsmm.M):
            b[:,n] += gmmhsmm.gmm_weights[m,n]*st.multivariate_normal.pdf(x, mean=gmmhsmm.gmm_mu[m,:,n], cov=gmmhsmm.gmm_cov[m,:,:,n])
    
    return(b)


def taskTimeEst(gmmhsmm,x,s,d): # Last observation acquired, failure state
    # initialize D
    D = np.zeros(3)
    
    b = likelihood(gmmhsmm, x)
    alpha, A, d_hat, lik = forward(gmmhsmm, b, d)
    
    delta_prob = np.zeros((x.shape[0],gmmhsmm.N))
    iter_count = 0
    
    # Current state estimation
    for t in range(x.shape[0]):
        if t==0:
            delta_prob[t,:] = gmmhsmm.pi*b[0,:]
            delta_prob[t,:] /= np.sum(delta_prob[t,:])
        else:
            delta_prob[t,:] = np.max((delta_prob[t-1,:]*A[t-1,:,:].T).T*b[t,:],axis=0)
            delta_prob[t,:] /= np.sum(delta_prob[t,:])
    
    delta_state = np.argmax(delta_prob[-1,:])
    #Fix update of D after first iteration
    while(delta_state != s and iter_count < gmmhsmm.N*2):
        print(iter_count)
        D_est = np.zeros(3)
        if iter_count==0:
            for n in range(gmmhsmm.N):
                D_est[1] += ((gmmhsmm.sd_mean[:,n]) - d_hat[-1,n])*delta_prob[-1,n]
                D_est[0] += ((gmmhsmm.sd_mean[:,n]) - 1.96*np.sqrt(gmmhsmm.sd_var[:,n]) - d_hat[-1,n])*delta_prob[-1,n]
                D_est[2] += ((gmmhsmm.sd_mean[:,n]) + 1.96*np.sqrt(gmmhsmm.sd_var[:,n]) - d_hat[-1,n])*delta_prob[-1,n]
        else:
            for n in range(gmmhsmm.N):
                D_est[1] += (gmmhsmm.sd_mean[:,n])*delta_prob[-1,n]
                D_est[0] += ((gmmhsmm.sd_mean[:,n]) - 1.96*np.sqrt(gmmhsmm.sd_var[:,n]))*delta_prob[-1,n]
                D_est[2] += ((gmmhsmm.sd_mean[:,n]) + 1.96*np.sqrt(gmmhsmm.sd_var[:,n]))*delta_prob[-1,n]
        D += D_est
        delta_prob[-1,:] = np.dot(gmmhsmm.A.T,delta_prob[-1,:].T)
        delta_prob[-1,:] /= np.sum(delta_prob[-1,:])
        
        delta_state = np.argmax(delta_prob[-1,:].T)
        
        iter_count += 1
    
    print(delta_state)
    return(D)


def gmmgamma(gmmhsmm, gamma, x):
    gmm_temp = np.zeros((x.shape[0], gmmhsmm.M, gmmhsmm.N))
    for n in range(gmmhsmm.N):
        for m in range(gmmhsmm.M):
            gmm_temp[:,m,n] = st.multivariate_normal.pdf(x, mean=gmmhsmm.gmm_mu[m,:,n], cov=gmmhsmm.gmm_cov[m,:,:,n])
    for t in range(x.shape[0]):
        gmm_temp[t,:,:] /= np.sum(gmm_temp[t,:,:], axis=0)
    gamma = gamma[:,np.newaxis,:]*gmm_temp
    return(gamma)


def train(gmmhsmm, x):
    for j in range(100):
        gmmhsmm_copy = copy.deepcopy(gmmhsmm)
        
        denom_e = np.zeros(gmmhsmm.sd_mean.shape)
        sd_mean_e = np.zeros(gmmhsmm.sd_mean.shape)
        sd_var_e = np.zeros(gmmhsmm.sd_mean.shape)
        gamma_sum_e = np.zeros((gmmhsmm.M, gmmhsmm.N))
        gmm_cov_e = np.zeros(gmmhsmm.gmm_cov.shape)
        gmm_mu_e = np.zeros((gmmhsmm.n_dims, gmmhsmm.M, gmmhsmm.N))
        
        xi_sum = np.zeros((gmmhsmm.N, gmmhsmm.N))
        
        
        
        for obs_i in x:
            b = likelihood(gmmhsmm, obs_i)
            alpha, A, d_hat, lik_old = forward(gmmhsmm, b, d)
            
            beta = backward(gmmhsmm, b, A)
        
            gamma_hmm = (alpha*beta)
            gamma_hmm /= np.sum(gamma_hmm, axis=1, keepdims=True)
        
            xi = np.zeros((alpha.shape[0]-1, gmmhsmm.N, gmmhsmm.N))
            for t in range(alpha.shape[0]-1):
                xi[t,:,:] = (alpha[t,:]*A[t,:,:].T).T*b[t+1,:]*beta[t+1,:]
        
            xi_sum = np.sum(xi, axis=0)
            
        #Update State Duration Parameters    
            for i in range(gmmhsmm.N-1):
                denominator = 0
                temp_sd_mu = 0
                temp_sd_var = 0
                for t in range(obs_i.shape[0]-1):
                    denominator += alpha[t,i]*np.sum((A[t,:,:]-A[t,:,:]*np.identity(gmmhsmm.N))[i,:]*b[t+1,:]*beta[t+1,:])
                    temp_sd_mu += alpha[t,i]*np.sum((A[t,:,:]-A[t,:,:]*np.identity(gmmhsmm.N))[i,:]*b[t+1,:]*beta[t+1,:])*d_hat[t,i]
                    temp_sd_var += alpha[t,i]*np.sum((A[t,:,:]-A[t,:,:]*np.identity(gmmhsmm.N))[i,:]*b[t+1,:]*beta[t+1,:])*np.power(d_hat[t,i]-gmmhsmm.sd_mean[0,i],2)
                
                
                sd_mean_e[0,i] += temp_sd_mu
                sd_var_e[0,i] += temp_sd_var
                denom_e[0,i] += denominator
            
            #Update GMM parameters
            #Update weights
            
            gamma_gmm = gmmgamma(gmmhsmm, gamma_hmm, obs_i)
            gamma_sum_e += np.sum(gamma_gmm, axis=0)
            
            #Update covs
            temp_cov = np.zeros((gmmhsmm.n_dims,gmmhsmm.n_dims))
            data_cov = obs_i[:,np.newaxis,:,np.newaxis]-gmmhsmm.gmm_mu[np.newaxis,:,:,:]
            
            for n in range(gmmhsmm.N):
                for m in range(gmmhsmm.M):
                    temp_cov = np.zeros((gmmhsmm.n_dims,gmmhsmm.n_dims))
                    for t in range(x.shape[0]):
                        temp_cov += (np.atleast_2d(data_cov[t,m,:,n])*np.atleast_2d(data_cov[t,m,:,n]).T)*gamma_gmm[t,m,n]
                    gmm_cov_e[m,:,:,n] += temp_cov
                    
            
            #Update means
            gmm_mu_e += np.sum(obs_i[:,:,np.newaxis, np.newaxis]*gamma_gmm[:,np.newaxis,:,:],axis=0)
            
            #Update HSMM parameters
            #gmmhsmm.A = (np.sum(xi, axis=0).T/np.sum(np.sum(xi, axis=0),axis=1)).T*(np.ones(gmmhsmm.A.shape)-np.diag(np.ones(gmmhsmm.N)))
            #gmmhsmm.A[-1,-1] = 1
            #gmmhsmm.A /= np.sum(gmmhsmm.A, axis=1, keepdims=True)
            
        for n in range(gmmhsmm.N-1):
            gmmhsmm.sd_mean[0,n] = sd_mean_e[0,n]/denom_e[0,n] 
            gmmhsmm.sd_var[0,n] = sd_var_e[0,n]/denom_e[0,n]
            for m in range(gmmhsmm.M):
                gmmhsmm.gmm_cov[m,:,:,n] = gmm_cov_e[m,:,:,n]/gamma_sum_e[m,n]
                gmmhsmm.gmm_cov[m,:,:,n] = gmmhsmm.gmm_cov[m,:,:,n] + np.eye(gmmhsmm.gmm_cov[m,:,:,n].shape[0])*0.1
                if np.linalg.matrix_rank(gmmhsmm.gmm_cov[m,:,:,n]) != gmmhsmm.N:
                    print("Covariance matrix is singular!")
                    gmmhsmm.gmm_cov[m,:,:,n] = gmmhsmm_copy.gmm_cov[m,:,:,n]
        
        gmmhsmm.gmm_mu = np.swapaxes(gmm_mu_e/gamma_sum_e[np.newaxis,:,:],0,1)
        gmmhsmm.gmm_weights = gamma_sum_e/np.sum(gamma_sum_e, axis=0, keepdims=True)
        
        
        gmmhsmm.gmm_mu[:,:,-1] = 10
        #Nan checking
        gmmhsmm.gmm_mu[np.isnan(gmmhsmm.gmm_mu)] = gmmhsmm_copy.gmm_mu[np.isnan(gmmhsmm.gmm_mu)]
        gmmhsmm.gmm_cov[np.isnan(gmmhsmm.gmm_cov)] = gmmhsmm_copy.gmm_cov[np.isnan(gmmhsmm.gmm_cov)]
        gmmhsmm.gmm_weights[np.isnan(gmmhsmm.gmm_weights)] = gmmhsmm_copy.gmm_weights[np.isnan(gmmhsmm.gmm_weights)]
            
        gmmhsmm.sd_mean[np.isnan(gmmhsmm.sd_mean)] = gmmhsmm_copy.sd_mean[np.isnan(gmmhsmm.sd_mean)]
        gmmhsmm.sd_var[np.isnan(gmmhsmm.sd_var)] = gmmhsmm_copy.sd_var[np.isnan(gmmhsmm.sd_var)]
        
#        test_A = (xi_sum.T/np.sum(xi_sum,axis=1)).T*(np.ones(gmmhsmm.A.shape)-np.diag(np.ones(gmmhsmm.N)))
#        test_A /= np.sum(test_A, axis=1, keepdims=True)
#        test_A[np.isnan(test_A)] = gmmhsmm.A[np.isnan(test_A)]
#        gmmhsmm.A = test_A
        
        lik_new = 0
        lik_old = 0
        
        for obs_j in x:
            b = likelihood(gmmhsmm, obs_j)
            alpha, A, d_hat, lik_temp = forward(gmmhsmm, b, d)
        
            lik_new += np.nan_to_num(lik_temp)
            
            b = likelihood(gmmhsmm_copy, obs_j)
            alpha, A, d_hat, lik_temp = forward(gmmhsmm_copy, b, d)
            
            lik_old += np.nan_to_num(lik_temp)
        
        
        print("Iteration "+str(j)+" likelihood: "+ str(lik_new) + str(lik_old))
        
        if (lik_old >= lik_new):
            gmmhsmm = gmmhsmm_copy
            print("Likelihood did not improve, model has converged")
            return(lik_old)
            
    return(lik_new)

class GMM_HSMM:
    def __init__(self, N=2, M=1, n_dims=1):
        # General parameters
        self.N = N
        self.M = M
        self.n_dims = n_dims
        
        # Markov chain parameters
        self.A = np.zeros((N,N))
        self.pi = np.zeros((1,N))
        
        # Observation parameters
        self.gmm_mu = np.zeros((M,n_dims,N))
        self.gmm_cov = np.zeros((M,n_dims,n_dims,N))
        self.gmm_weights = np.zeros((M,N))
        
        # State duration parameters
        self.sd_mean = np.zeros((1,N))
        self.sd_var = np.zeros((1,N))


#Read in data
obs_sd = pd.read_csv('woz_final.csv', header=0)
#x accel, y accel, z accel, lane dev, vehical speed, accel pedal, angular vel, steering angle, steering rate, task time
obs_sim = np.loadtxt('CleanData/train_csv/PID113B.csv', delimiter=',', skiprows=1)

label = obs_sim[:,9]
x = obs_sim[:,[1,6]]

task_index = segmentData(label)

obs_list = np.array([x[i:j+60,:] for i,j in zip(task_index[::2],task_index[1::2])])
label_list = np.array([label[i:j+60] for i,j in zip(task_index[::2],task_index[1::2])])

#Normalize variance
obs_std = np.std(np.vstack(obs_list),axis=0)
obs_mean = np.mean(np.vstack(obs_list),axis=0)
for i in range(len(obs_list)):
    obs_list[i] = (obs_list[i] -obs_mean[np.newaxis,:])/obs_std[np.newaxis,:]


#Numpy random seed
np.random.seed(213)

best_loss_huber = 9999999

T = 0
for i in obs_list:
    T += i.shape[0]


n_dims = 2
N = 4
M = 2


sd_times = np.floor(np.array(obs_sd[obs_sd['Recording'] == '113b']['task_time'])*59)
target_times = np.array(obs_sd[obs_sd['Recording'] == '113b']['task_time'])

d = np.zeros(N)
d[0] = 1.0/59.0

temp = GMM_HSMM(N,M,n_dims)

if N==3:
    temp.pi = np.array([[1,0,0]])
    temp.A = np.array([[0,1,0],[0,0,1],[0,0,1]])
elif N==4:
    temp.pi = np.array([[1,0,0,0]])
    temp.A = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,1]])
else:
    temp.pi = np.array([[1,0,0,0,0]])
    temp.A = np.array([[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1],[0,0,0,0,1]])

cv_loss_huber = 0
    
for _ in range(40):
    obs_gmm = GaussianMixture(n_components=M, covariance_type='full')
    for i in range(N):
        obs_gmm.fit(obs_list[np.random.randint(0,len(obs_list))])
        temp.gmm_mu[:,:,i] = obs_gmm.means_
        temp.gmm_cov[:,:,:,i] = obs_gmm.covariances_
        temp.gmm_weights[:,i] = obs_gmm.weights_

    obs_gamma = st.norm.fit(obs_sd[obs_sd['Recording']=="113b"]['task_time'])
    obs_gamma_div = np.random.randint(0,100,size=N-1)

    for n in range(N):
        if (n==N-1):
            temp.sd_mean[:,n] = 1
            temp.sd_var[:,n] = 1
        else:
            temp.sd_mean[:,n] = obs_gamma[0]*(obs_gamma_div[n]/np.sum(obs_gamma_div))
            temp.sd_var[:,n] = obs_gamma[1]
            
    current_lik = train(temp, obs_list)

    current_loss_huber = 0
    for obs_i in range(len(sd_times)):
        times = np.concatenate(( np.floor( np.arange(0, sd_times[obs_i], sd_times[obs_i]/10)), np.atleast_1d(sd_times[obs_i])))
        for t in times:
            pred_temp = taskTimeEst(temp, obs_list[obs_i][0:int(t+1),:], temp.N-1, d)[1]
            if np.abs((sd_times[obs_i]/59.0 - t/59.0) - pred_temp) <= 1:
                current_loss_huber += 0.5*np.power((sd_times[obs_i]/59.0 - t/59.0) - pred_temp, 2)
            else:
                current_loss_huber += np.abs((sd_times[obs_i]/59.0 - t/59.0) - pred_temp) - 0.5
            
    current_loss_huber /= 11*len(sd_times)
        
    if current_loss_huber < best_loss_huber:
        best_model_huber = copy.deepcopy(temp)
        best_loss_huber = current_loss_huber    
        
#pickle object
pickle.dump(best_model_huber, open("D:/OneDrive/Documents/PhD/PhD Thesis/papers/task init time/final models/113_cv.p", "wb"))

#best_model_list = pickle.load( open("D:/OneDrive/Documents/PhD/PhD Thesis/papers/task init time/models/102.p", "rb" ) )