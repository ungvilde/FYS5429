import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from naive_bayes import NaiveBayesRegression
from metrics import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# dataset = 'hc'
dataset = 'm1'
tuning_form ='quadratic'

with open('data/' + dataset + '_neural_data.pickle','rb') as f:
    neural_data,pos_binned=pickle.load(f,encoding='latin1') 

if dataset=='hc':
    bins_before=4 
    bins_current=1 
    bins_after=5 

if dataset=='m1':
    bins_before=2 
    bins_current=1 
    bins_after=0 

#Remove neurons with too few spikes in CA1 data
if dataset=='hc':
    nd_sum=np.nansum(neural_data,axis=0)
    rmv_nrn=np.where(nd_sum<100)
    neural_data=np.delete(neural_data,rmv_nrn,1)

X=neural_data
y=pos_binned
N=bins_before+bins_current+bins_after # num. timebins

if dataset=='hc':
    #Remove time bins with no output 
    rmv_time=np.where(np.isnan(y[:,0]) | np.isnan(y[:,1]))
    X=np.delete(X,rmv_time,0)
    y=np.delete(y,rmv_time,0)

if dataset=='hc':
    X=X[:int(.8*X.shape[0]),:]
    y=y[:int(.8*y.shape[0]),:]

K=X.shape[0] # Total time bins

# Training/testing/validation sections
valid_range_all=[
    [0,.1], [.1,.2], [.2,.3], [.3,.4], [.4,.5], [.5,.6], [.6,.7], [.7,.8], [.8,.9], [.9,1]
    ]
testing_range_all=[
    [.1,.2], [.2,.3], [.3,.4], [.4,.5], [.5,.6], [.6,.7], [.7,.8], [.8,.9], [.9,1], [0,.1]
    ]
training_range_all=[
    [[.2,1]],
    [[0,.1],[.3,1]],
    [[0,.2],[.4,1]],
    [[0,.3],[.5,1]],
    [[0,.4],[.6,1]],
    [[0,.5],[.7,1]],
    [[0,.6],[.8,1]],
    [[0,.7],[.9,1]],
    [[0,.8]],[[.1,.9]]
    ]

num_folds=len(valid_range_all) 

# For storing the R2 values
mean_R2=np.empty(num_folds)

for i in range(num_folds):

    testing_range = testing_range_all[i]
    testing_set = np.arange(
        np.int(np.round(testing_range[0]*K)) + bins_before, #with bins_before to avoid overlap
        np.int(np.round(testing_range[1]*K)) - bins_after
        )

    valid_range = valid_range_all[i]
    valid_set = np.arange(
        np.int(np.round(valid_range[0]*K))+bins_before,
        np.int(np.round(valid_range[1]*K))-bins_after
        )

    training_ranges=training_range_all[i]
    for j in range(len(training_ranges)): 
        training_range=training_ranges[j]
        if j==0:
            training_set=np.arange(
                np.int(np.round(training_range[0]*K)) + bins_before,
                np.int(np.round(training_range[1]*K)) - bins_after
                )
        if j==1: # concatentate to first part of training data
            training_set_temp = np.arange(
                np.int(np.round(training_range[0]*K))+bins_before,
                np.int(np.round(training_range[1]*K))-bins_after
                )
            training_set=np.concatenate((training_set,training_set_temp),axis=0)
                
    X_train=X[training_set,:]
    y_train=y[training_set,:]
    
    X_test=X[testing_set,:]
    y_test=y[testing_set,:]

    X_valid=X[valid_set,:]
    y_valid=y[valid_set,:]

    
    # Initialise neural data in Naive bayes format
    num_nrns=X_train.shape[1]
    X_b_train=np.empty([X_train.shape[0]-N+1,num_nrns])
    X_b_valid=np.empty([X_valid.shape[0]-N+1,num_nrns])
    X_b_test=np.empty([X_test.shape[0]-N+1,num_nrns])
    
    # Get the total number of spikes across time bins
    for k in range(num_nrns):
        X_b_train[:,k] = N*np.convolve(X_train[:,k], np.ones((N,))/N, mode='valid') #Convolving w/ ones is a sum across those N bins
        X_b_valid[:,k]=N*np.convolve(X_valid[:,k], np.ones((N,))/N, mode='valid')
        X_b_test[:,k]=N*np.convolve(X_test[:,k], np.ones((N,))/N, mode='valid')

    X_b_train=X_b_train.astype(int)
    X_b_valid=X_b_valid.astype(int)
    X_b_test=X_b_test.astype(int)

    if bins_before>0 and bins_after>0:
        y_train=y_train[bins_before:-bins_after,:]
        y_valid=y_valid[bins_before:-bins_after,:]
        y_test=y_test[bins_before:-bins_after,:]

    if bins_before>0 and bins_after==0:
        y_train=y_train[bins_before:,:]
        y_valid=y_valid[bins_before:,:]
        y_test=y_test[bins_before:,:]
    
    if dataset=='hc':
        res=100
    if dataset=='m1' or dataset=='s1':
        res=50    
    
    model_nb=NaiveBayesRegression(encoding_model=tuning_form,res=res) 
    model_nb.fit(X_b_train,y_train)

    y_test_predicted_nb=model_nb.predict(X_b_test,y_test)   
    mean_R2[i]=np.mean(get_R2(y_test,y_test_predicted_nb))    
    R2s_nb=get_R2(y_test,y_test_predicted_nb)
    print('R2s:', R2s_nb)
    
    
    with open('results/results_nb3_prior_'+tuning_form+'_'+dataset+'.pickle','wb') as f:
        pickle.dump([mean_R2],f)                    