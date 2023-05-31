import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

from simple_rnn import SimpleRNNRegression
from lstm import LSTMRegression
from data_preprocessing import *
from metrics import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

sns.set_theme("notebook", "whitegrid", palette="colorblind")

cm=1/2.54
params = {
    'legend.fontsize': 9,
    'font.size': 9,
    'figure.figsize': (8.647*cm, 12.0*cm), # figsize for two-column latex doc
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'lines.markersize': 3.0,
    'lines.linewidth': 1.5,
    }

plt.rcParams.update(params)


dataset = 'm1'
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

#Remove neurons with too few spikes in CA1 dataset
if dataset=='hc':
    nd_sum=np.nansum(neural_data,axis=0)
    rmv_nrn=np.where(nd_sum<100)
    neural_data=np.delete(neural_data,rmv_nrn,1)

X = get_spikes_with_history(neural_data, bins_before, bins_after, bins_current)
y = output_binned

if dataset=='hc':
    #Remove time bins with no output (y value)
    rmv_time=np.where(np.isnan(y[:,0]) | np.isnan(y[:,1]))
    X=np.delete(X,rmv_time,0)
    y=np.delete(y,rmv_time,0)

    # only inlcude data with activity
    X=X[:int(.8*X.shape[0]),:,:]
    y=y[:int(.8*y.shape[0]),:]

training_range=[0, 0.7]
testing_range=[0.7, 1.0]

num_examples = X.shape[0]

training_set = np.arange(
    np.int(np.round(training_range[0]*num_examples)) + bins_before,
    np.int(np.round(training_range[1]*num_examples)) - bins_after
    )
testing_set = np.arange(
    np.int(np.round(testing_range[0]*num_examples)) + bins_before,
    np.int(np.round(testing_range[1]*num_examples)) - bins_after
    )

X_train=X[training_set,:,:]
y_train=y[training_set,:]

X_test=X[testing_set,:,:]
y_test=y[testing_set,:]

if bins_before>0 and bins_after>0:
    y_test_nb=y_test[bins_before:-bins_after,:]
    
if bins_before>0 and bins_after==0:
    y_test_nb=y_test[bins_before:,:]

# center and scale data
X_train_mean = np.nanmean(X,axis=0) #Mean of training data
X_train_std = np.where(np.nanstd(X,axis=0)==0, 1, np.nanstd(X,axis=0)) #Std of training data, avoid dividing by zeros
X_train = (X_train - X_train_mean)/X_train_std 
X_test = (X_test - X_train_mean)/X_train_std 

y_train_mean=np.nanmean(y,axis=0) #Mean of training data outputs
y_train = y_train - y_train_mean #Zero-center training output
y_test = y_test - y_train_mean

with open('results/rnn_fitted_'+dataset+'.pickle','rb') as f:
    model_rnn = pickle.load(f)

with open('results/lstm_fitted_'+dataset+'.pickle','rb') as f:
    model_lstm = pickle.load(f)

with open('results/naiveBayes_quadratic_predicted_'+dataset+'.pickle','rb') as f:
    y_pred_naiveBayes = pickle.load(f)

y_pred_rnn = model_rnn.predict(X_test=X_test)
y_pred_lstm = model_lstm.predict(X_test=X_test)

print("R2s RNN = ",get_R2(y_test, y_pred_rnn))
print("median error of RNN = ",np.median(np.abs(y_test - y_pred_rnn)))

print("R2s LSTM = ",get_R2(y_test, y_pred_lstm))
print("median error of LSTM = ",np.median(np.abs(y_test - y_pred_lstm)))

# print("R2s naive Bayes = ",get_R2(y_test_nb, y_pred_naiveBayes))


fig, ax = plt.subplots(ncols=1, nrows=3, sharex=True, sharey=True)


if dataset == 'm1':
    dt=0.05
    start_bin=1000
    plot_bins=200
    y_label = "$y$ velocity [cm/s]"

if dataset=='hc':
    dt=0.2
    start_bin=3500
    start_bin=4500
    plot_bins=750
    y_label = "$y$ position [cm]"

end_bin=start_bin+plot_bins
time = np.arange(0, plot_bins*dt, step=dt)

ax[0].set_title("Simple RNN")
ax[0].plot(time,y_test[start_bin:end_bin, 0] + y_train_mean[0], '-k',label = "Ground truth")
ax[0].plot(time,y_pred_rnn[start_bin:end_bin, 0] + y_train_mean[0], '-r', label = "Predicted")

ax[1].set_title("LSTM")
ax[1].plot(time,y_test[start_bin:end_bin, 0] + y_train_mean[0], '-k',label = "Ground truth")
ax[1].plot(time,y_pred_lstm[start_bin:end_bin, 0] + y_train_mean[0], '-b', label = "Predicted")
ax[1].set_xlabel("Time [s]")
ax[1].set_ylabel(y_label)

ax[2].set_title("Naive Bayes")
ax[2].plot(time,y_test_nb[(start_bin-bins_before):(end_bin-bins_before), 0], '-k',label = "Ground truth")
ax[2].plot(time,y_pred_naiveBayes[(start_bin-bins_before):(end_bin-bins_before), 0], '-g', label = "Predicted")
ax[2].set_xlabel("Time [s]")

plt.tight_layout()
plt.savefig('figs/predicted_vals_x_'+dataset+'.pdf')

###
fig, ax = plt.subplots(ncols=1, nrows=3, sharex=True, sharey=True)


if dataset == 'm1':
    dt=0.05
    start_bin=1000
    plot_bins=300
    y_label = "$y$ velocity [cm/s]"

if dataset=='hc':
    dt=0.2
    start_bin=3500
    start_bin=4500
    plot_bins=75
    y_label = "$y$ position [cm]"

end_bin=start_bin+plot_bins
time = np.arange(0, plot_bins*dt, step=dt)

ax[0].set_title("Simple RNN")
ax[0].plot(time,y_test[start_bin:end_bin, 1] + y_train_mean[1], '-k',label = "Ground truth")
ax[0].plot(time,y_pred_rnn[start_bin:end_bin, 1] + y_train_mean[1], '-r', label = "Predicted")

ax[1].set_title("LSTM")
ax[1].plot(time,y_test[start_bin:end_bin, 1] + y_train_mean[1], '-k',label = "Ground truth")
ax[1].plot(time,y_pred_lstm[start_bin:end_bin, 1] + y_train_mean[1], '-b', label = "Predicted")
ax[1].set_xlabel("Time [s]")
ax[1].set_ylabel(y_label)

ax[2].set_title("Naive Bayes")
ax[2].plot(time,y_test_nb[(start_bin-bins_before):(end_bin-bins_before), 1], '-k',label = "Ground truth")
ax[2].plot(time,y_pred_naiveBayes[(start_bin-bins_before):(end_bin-bins_before), 1], '-g', label = "Predicted")
ax[2].set_xlabel("Time [s]")

plt.tight_layout()
plt.savefig('figs/predicted_vals_y_'+dataset+'.pdf')

####


if dataset == 'm1':
    dt=0.05
    start_bin=100
    plot_bins=300
    y_label = "$y$ velocity [cm/s]"

if dataset=='hc':
    dt=0.2
    start_bin=2000
    plot_bins=75
    y_label = "$y$ position [cm]"

end_bin=start_bin+plot_bins

fig, ax = plt.subplots(ncols=3, nrows=1, sharex=True, sharey=True, figsize=(2*8.647*cm, 8.0*cm))
ax[0].set_aspect('equal', adjustable='box')
ax[0].set_title("Simple RNN")
ax[0].plot(y_test[start_bin:end_bin, 0] + y_train_mean[0], y_test[start_bin:end_bin, 1] + y_train_mean[1], '-k',label = "Ground truth")
ax[0].plot(y_pred_rnn[start_bin:end_bin, 0] + y_train_mean[0], y_pred_rnn[start_bin:end_bin, 1] + y_train_mean[1], '-r', label = "Predicted")
ax[0].set_ylabel("$y$-values")

ax[1].set_aspect('equal', adjustable='box')
ax[1].set_title("LSTM")
ax[1].plot(y_test[start_bin:end_bin, 0] + y_train_mean[0],y_test[start_bin:end_bin, 1] + y_train_mean[1], '-k',label = "Ground truth")
ax[1].plot(y_pred_lstm[start_bin:end_bin, 0] + y_train_mean[0],y_pred_lstm[start_bin:end_bin, 1] + y_train_mean[1], '-b', label = "Predicted")
ax[1].set_xlabel("$x$-values")

ax[2].set_aspect('equal', adjustable='box')
ax[2].set_title("Naive Bayes")
ax[2].plot(y_test_nb[(start_bin-bins_before):(end_bin-bins_before), 0], y_test_nb[(start_bin-bins_before):(end_bin-bins_before), 1], '-k',label = "Ground truth")
ax[2].plot(y_pred_naiveBayes[(start_bin-bins_before):(end_bin-bins_before), 0], y_pred_naiveBayes[(start_bin-bins_before):(end_bin-bins_before), 1], '-g', label = "Predicted")

plt.tight_layout()
plt.savefig('figs/predicted_vals_xy_'+dataset+'.pdf')

# plt.figure()
# plt.plot(y_test[:, 0] + y_train_mean[0], y_test[:, 1] + y_train_mean[1], '-k',label = "Ground truth")
# plt.axis('equal')
# plt.show()