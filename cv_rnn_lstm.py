import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from bayes_opt import BayesianOptimization

from simple_rnn import SimpleRNNRegression
from lstm import LSTMRegression
from data_preprocessing import *
from metrics import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# dataset = 'm1'
dataset = 'hc'

with open('data/' + dataset + '_neural_data.pickle','rb') as f:
    neural_data, output_binned=pickle.load(f)

if dataset == 'm1':
    bins_before=13 
    bins_current=1
    bins_after=0

if dataset == 'hc':
    bins_before=4 
    bins_current=1 
    bins_after=5

#Remove neurons with too few spikes 
if dataset=='hc':
    nd_sum=np.nansum(neural_data,axis=0)
    rmv_nrn=np.where(nd_sum<100)
    neural_data=np.delete(neural_data,rmv_nrn,1)

X=get_spikes_with_history(neural_data, bins_before, bins_after, bins_current)
y=output_binned

if dataset=='hc':
    #Remove time bins with no output 
    rmv_time=np.where(np.isnan(y[:,0]) | np.isnan(y[:,1]))
    X=np.delete(X,rmv_time,0)
    X_flat=np.delete(X_flat,rmv_time,0)
    y=np.delete(y,rmv_time,0)

    # only inlcude data with activity
    X=X[:int(.8*X.shape[0]),:,:]
    X_flat=X_flat[:int(.8*X_flat.shape[0]),:]
    y=y[:int(.8*y.shape[0]),:]

valid_range_all=[
    [0,.1],[.1,.2],[.2,.3],[.3,.4],[.4,.5],[.5,.6],[.6,.7],[.7,.8],[.8,.9],[.9,1]
    ]
testing_range_all=[
    [.1,.2],[.2,.3],[.3,.4],[.4,.5],[.5,.6],[.6,.7],[.7,.8],[.8,.9],[.9,1],[0,.1]
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
    [[0,.8]],[[.2,.9]]
    ]
num_folds=len(valid_range_all) 

# Lists for storing results
r2_scores_lstm = [] 
r2_scores_rnn = []

K=X.shape[0] #number of time bins total

for i in range(num_folds): #Loop through the folds
    
    print("Fold ", i)

    testing_range=testing_range_all[i]
    testing_set=np.arange(
        np.int(np.round(testing_range[0]*K))+bins_before,
        np.int(np.round(testing_range[1]*K))-bins_after
        )

    valid_range=valid_range_all[i]
    valid_set=np.arange(
        np.int(np.round(valid_range[0]*K))+bins_before,
        np.int(np.round(valid_range[1]*K))-bins_after
        )

    training_ranges=training_range_all[i]
    for j in range(len(training_ranges)): 
        training_range=training_ranges[j]
        if j==0: 
            training_set=np.arange(
                np.int(np.round(training_range[0]*K))+bins_before,
                np.int(np.round(training_range[1]*K))-bins_after
                )
        if j==1: 
            training_set_temp=np.arange(
                np.int(np.round(training_range[0]*K))+bins_before,
                np.int(np.round(training_range[1]*K))-bins_after
                )
            training_set=np.concatenate((training_set,training_set_temp),axis=0)
                
    X_train=X[training_set,:,:]
    y_train=y[training_set,:]
    
    X_test=X[testing_set,:,:]
    y_test=y[testing_set,:]

    X_valid=X[valid_set,:,:]
    y_valid=y[valid_set,:]
    
    # standardise data
    X_train_mean=np.nanmean(X_train,axis=0) 
    X_train_std=np.nanstd(X_train,axis=0) 
    X_train=(X_train-X_train_mean)/X_train_std 
    X_test=(X_test-X_train_mean)/X_train_std
    X_valid=(X_valid-X_train_mean)/X_train_std 

    # center outputs
    y_train_mean=np.nanmean(y_train,axis=0) 
    y_train=y_train-y_train_mean 
    y_test=y_test-y_train_mean 
    y_valid=y_valid-y_train_mean 

    hyperparam_space = {
        'units': [100, 500], #range in units
        'num_epochs': [5, 10] #range in epochs
    }

    def evaluate_rnn(units, num_epochs):
        units = int(units)
        num_epochs = int(num_epochs)
        model = SimpleRNNRegression(units=units, num_epochs=num_epochs, verbose=0)
        model.fit(X_train=X_train, y_train=y_train)
        y_pred = model.predict(X_test=X_valid)
        return r2_score(y_true=y_valid, y_pred=y_pred)

    print("RNN:")
    optimize_rnn = BayesianOptimization(evaluate_rnn, hyperparam_space, verbose=1)
    optimize_rnn.maximize(init_points=10, n_iter=10)
    
    units = np.int(optimize_rnn.max["params"]["units"])
    num_epochs = np.int(optimize_rnn.max["params"]["num_epochs"])

    model_rnn = SimpleRNNRegression(units=units, num_epochs=num_epochs, verbose=0)
    model_rnn.fit(X_train=X_train, y_train=y_train)
    y_pred_rnn = model_rnn.predict(X_test=X_test)
    r2_rnn = r2_score(y_true=y_test, y_pred=y_pred_rnn)
    r2_scores_rnn.append(r2_rnn) #save test r2 score
    print("R2 (RNN) = ",r2_rnn)

    print("LSTM:")
    def evaluate_lstm(units, num_epochs):
        units = int(units)
        num_epochs = int(num_epochs)
        model = LSTMRegression(units=units, num_epochs=num_epochs, verbose=0)
        model.fit(X_train=X_train, y_train=y_train)
        y_pred = model.predict(X_test=X_valid)
        return r2_score(y_true=y_valid, y_pred=y_pred)

    optimize_lstm = BayesianOptimization(evaluate_lstm, hyperparam_space, verbose=1)
    optimize_lstm.maximize(init_points=10, n_iter=10)

    units = np.int(optimize_lstm.max["params"]["units"])
    num_epochs = np.int(optimize_lstm.max["params"]["num_epochs"])

    model_lstm = LSTMRegression(units=units, num_epochs=num_epochs, verbose=0)
    model_lstm.fit(X_train=X_train, y_train=y_train)
    y_pred_lstm = model_lstm.predict(X_test=X_test)
    r2_lstm = r2_score(y_true=y_test, y_pred=y_pred_lstm)
    r2_scores_lstm.append(r2_lstm) 
    print("R2 (LSTM) = ", r2_lstm)

with open('results/rnn_results_'+dataset+'.pickle','wb') as f:
    pickle.dump([r2_scores_rnn], f)

with open('results/lstm_results_'+dataset+'.pickle','wb') as f:
    pickle.dump([r2_scores_lstm], f)


