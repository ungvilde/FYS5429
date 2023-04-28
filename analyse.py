import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from simple_rnn import SimpleRNNRegression

from data_preprocessing import *
from metrics import *

datafile = 'm1'

with open('data/' + datafile + '_neural_data.pickle','rb') as f:
    neural_data, vels_binned=pickle.load(f)

bins_before=13 #How many bins of neural data prior to the output are used for decoding
bins_current=1 #Whether to use concurrent time bin of neural data
bins_after=0

X=get_spikes_with_history(neural_data, bins_before, bins_after, bins_current)
y=vels_binned

X = X[(bins_before+2):] # first 14 slices do not have any values because of history effects
y = y[(bins_before+2):]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=543)

# Scale the data
X_train_mean=np.nanmean(X_train,axis=0) #Mean of training data
X_train_std=np.nanstd(X_train,axis=0) #Stdev of training data
X_train=(X_train-X_train_mean)/X_train_std #Z-score training data
X_test=(X_test-X_train_mean)/X_train_std #Preprocess testing data in same manner as training data

print("X_train shape=",X_train.shape)
print("y_train shape=",y_train.shape)

print("X_test shape=",X_test.shape)
print("y_test shape=",y_test.shape)

rnn = SimpleRNNRegression(units=100)

rnn.fit(X_train=X_train, y_train=y_train)
y_pred = rnn.predict(X_test=X_test)

print("y_pred shape = ", y_pred.shape)

R2 = r2_score(y_true = y_test, y_pred = y_pred)

print("R2 = ", R2)