import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from naive_bayes import NaiveBayesRegression
from metrics import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

dataset = 'hc'

with open('data/' + dataset + '_neural_data.pickle','rb') as f:
    neural_data,pos_binned=pickle.load(f,encoding='latin1') 

if dataset=='hc':
    bins_before=4 #How many bins of neural data prior to the output are used for decoding
    bins_current=1 #Whether to use concurrent time bin of neural data
    bins_after=5 #How many bins of neural data after (and including) the output are used for decoding

#Remove neurons with too few spikes in HC dataset
if dataset=='hc':
    nd_sum=np.nansum(neural_data,axis=0)
    rmv_nrn=np.where(nd_sum<100)
    neural_data=np.delete(neural_data,rmv_nrn,1)

X=neural_data
y=pos_binned
N=bins_before+bins_current+bins_after # num. timebins

if dataset=='hc':
    #Remove time bins with no output (y value)
    rmv_time=np.where(np.isnan(y[:,0]) | np.isnan(y[:,1]))
    X=np.delete(X,rmv_time,0)
    y=np.delete(y,rmv_time,0)

# if dataset=='hc':
#     X=X[:int(.8*X.shape[0]),:]
#     y=y[:int(.8*y.shape[0]),:]


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=543)


#Number of examples after taking into account bins removed for lag alignment
num_examples=X.shape[0]

#Set what part of data should be part of the training/testing/validation sets

training_range=[0, 0.5]
valid_range=[0.5,0.65]
testing_range=[0.65, 0.8]

#Note that each range has a buffer of"bins_before" bins at the beginning, and "bins_after" bins at the end
#This makes it so that the different sets don't include overlapping neural data
training_set=np.arange(np.int(np.round(training_range[0]*num_examples))+bins_before,np.int(np.round(training_range[1]*num_examples))-bins_after)
testing_set=np.arange(np.int(np.round(testing_range[0]*num_examples))+bins_before,np.int(np.round(testing_range[1]*num_examples))-bins_after)
valid_set=np.arange(np.int(np.round(valid_range[0]*num_examples))+bins_before,np.int(np.round(valid_range[1]*num_examples))-bins_after)

#Get training data
X_train=X[training_set,:]
y_train=y[training_set,:]

#Get testing data
X_test=X[testing_set,:]
y_test=y[testing_set,:]

#Get validation data
X_valid=X[valid_set,:]
y_valid=y[valid_set,:]


#Initialize matrices for neural data in Naive bayes format
num_nrns=X_train.shape[1]
X_b_train=np.empty([X_train.shape[0]-N+1,num_nrns])
X_b_valid=np.empty([X_valid.shape[0]-N+1,num_nrns])
X_b_test=np.empty([X_test.shape[0]-N+1,num_nrns])

#Below assumes that bins_current=1 (otherwise alignment will be off by 1 between the spikes and outputs)

#For all neurons, within all the bins being used, get the total number of spikes (sum across all those bins)
#Do this for the training/validation/testing sets
for i in range(num_nrns):
    X_b_train[:,i]=N*np.convolve(X_train[:,i], np.ones((N,))/N, mode='valid') #Convolving w/ ones is a sum across those N bins
    X_b_valid[:,i]=N*np.convolve(X_valid[:,i], np.ones((N,))/N, mode='valid')
    X_b_test[:,i]=N*np.convolve(X_test[:,i], np.ones((N,))/N, mode='valid')

#Make integer format
X_b_train=X_b_train.astype(int)
X_b_valid=X_b_valid.astype(int)
X_b_test=X_b_test.astype(int)

#Make y's aligned w/ X's
#e.g. we have to remove the first y if we are using 1 bin before, and have to remove the last y if we are using 1 bin after
if bins_before>0 and bins_after>0:
    y_train=y_train[bins_before:-bins_after,:]
    y_valid=y_valid[bins_before:-bins_after,:]
    y_test=y_test[bins_before:-bins_after,:]
    
if bins_before>0 and bins_after==0:
    y_train=y_train[bins_before:,:]
    y_valid=y_valid[bins_before:,:]
    y_test=y_test[bins_before:,:]


# #Initialize matrices for neural data in Naive bayes format
# num_nrns=X_train.shape[1] #num. neurons
# X_b_train=np.empty([X_train.shape[0]-N+1, num_nrns])
# X_b_test=np.empty([X_test.shape[0]-N+1, num_nrns])

# #For all neurons, within all the bins being used, get the total number of spikes (sum across all those bins)
# #Do this for the training and testing sets
# for k in range(num_nrns):
#     X_b_train[:,k]=N*np.convolve(X_train[:,k], np.ones((N,))/N, mode='valid') #Convolving w/ ones is a sum across those N bins
#     X_b_test[:,k]=N*np.convolve(X_test[:,k], np.ones((N,))/N, mode='valid')

# #Make integer format
# X_b_train=X_b_train.astype(int)
# X_b_test=X_b_test.astype(int)

# #remove rows without previous bins
# if bins_before>0 and bins_after>0:
#     y_train=y_train[bins_before:-bins_after,:]
#     y_test=y_test[bins_before:-bins_after,:]

# if bins_before>0 and bins_after==0:
#     y_train=y_train[bins_before:,:]
#     y_test=y_test[bins_before:,:]

# print("num neurons =", num_nrns)
# print("X_b_train.shape = ",X_b_train.shape)
# print("y_train.shape = ", y_train.shape)

# print("X_b_test.shape = ",X_b_test.shape)
# print("y_test.shape = ", y_test.shape)

if dataset == 'hc':
    res=100 #resolution of target data point grid

model_nb=NaiveBayesRegression(encoding_model='quadratic',res=res)
model_nb.fit(X_b_train, y_train)
y_valid_predicted=model_nb.predict(X_b_valid,y_valid)
#Get metric of fit
R2_nb=get_R2(y_valid,y_valid_predicted)
print(R2_nb)
#Make example plot
plt.plot(y_valid[2000:2500,1])
plt.plot(y_valid_predicted[2000:2500,1])
plt.show()
# y_pred = model.predict(X_b_test, y_test)
# R2 = get_R2(y_test=y_test, y_test_pred=y_pred)
# print("R2 score = ", np.mean(R2))

# # plt.plot(y_pred[:100, 0])
# plt.plot(y[:1000, 0], y[:1000, 1], label="True")
# plt.axis('square')
# plt.legend()
# plt.xlabel("$x$ values")
# plt.ylabel("$y$ values")
# plt.show()