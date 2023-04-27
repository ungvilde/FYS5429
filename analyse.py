import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from simple_rnn import SimpleRNNRegression

from data_preprocessing import *

datafile = 's1'

with open('data/' + datafile + '_neural_data.pickle','rb') as f:
    neural_data, vels_binned=pickle.load(f)

bins_before=13 #How many bins of neural data prior to the output are used for decoding
bins_current=1 #Whether to use concurrent time bin of neural data
bins_after=0

X=get_spikes_with_history(neural_data, bins_before, bins_after, bins_current)
y=vels_binned

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

print("X_train shape=",X_train.shape)
print("y_train shape=",y_train.shape)

print("X_test shape=",X_test.shape)
print("y_test shape=",y_test.shape)

rnn = SimpleRNNRegression()

rnn.fit(X_train=X_train, y_train=y_train)