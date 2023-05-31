#Used for naive bayes decoder
import statsmodels.api as sm
import math

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.stats import norm
from scipy.spatial.distance import cdist

import numpy as np

# fit generalised linear model of tuning curve
def glm_run(Xr, Yr, X_range):
    X2 = sm.add_constant(Xr) #Include intercept

    poiss_model = sm.GLM(Yr, X2, family=sm.families.Poisson())
    try:
        glm_results = poiss_model.fit()
        Y_range=glm_results.predict(sm.add_constant(X_range))
    except:
        Y_range=np.mean(Yr)*np.ones([X_range.shape[0],1])
        
    return Y_range


class NaiveBayesRegression(object):

    """
    Class for the Naive Bayes Decoder

    Parameters
    ----------
    encoding_model: string, default='quadratic'
        what encoding model is used

    res:int, default=100
        resolution of predicted values
        This is the number of bins to divide the outputs into (going from minimum to maximum)
        larger values will make decoding slower
    """

    def __init__(self,encoding_model='quadratic',res=100):
        self.encoding_model=encoding_model
        self.res=res
        return

    def fit(self,X_b_train,y_train):

        """
        Train Naive Bayes Decoder

        Parameters
        ----------
        X_b_train: numpy 2d array of shape [n_samples,n_neurons]
            This is the neural training data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted (training data)
        """

        # For modelling tuning curve:
        # Find range for x and y (position/velocity) values
        input_x_range=np.arange(
            np.min(y_train[:,0]),
            np.max(y_train[:,0])+.01,
            np.round((np.max(y_train[:,0])-np.min(y_train[:,0]))/self.res)
            )
        input_y_range=np.arange(
            np.min(y_train[:,1]),
            np.max(y_train[:,1])+.01,
            np.round((np.max(y_train[:,1])-np.min(y_train[:,1]))/self.res))

        input_mat=np.meshgrid(input_x_range,input_y_range)
        xs=np.reshape(input_mat[0],[input_x_range.shape[0]*input_y_range.shape[0],1])
        ys=np.reshape(input_mat[1],[input_x_range.shape[0]*input_y_range.shape[0],1])
        input_xy=np.concatenate((xs,ys),axis=1)

        #If quadratic model:
        if self.encoding_model=='quadratic':
            input_xy_modified=np.empty([input_xy.shape[0],5])
            input_xy_modified[:,0]=input_xy[:,0]**2
            input_xy_modified[:,1]=input_xy[:,0]
            input_xy_modified[:,2]=input_xy[:,1]**2
            input_xy_modified[:,3]=input_xy[:,1]
            input_xy_modified[:,4]=input_xy[:,0]*input_xy[:,1]
            y_train_modified=np.empty([y_train.shape[0],5])
            y_train_modified[:,0]=y_train[:,0]**2
            y_train_modified[:,1]=y_train[:,0]
            y_train_modified[:,2]=y_train[:,1]**2
            y_train_modified[:,3]=y_train[:,1]
            y_train_modified[:,4]=y_train[:,0]*y_train[:,1]

        num_nrns=X_b_train.shape[1] 
        tuning_all=np.zeros([num_nrns,input_xy.shape[0]]) # stores tuning curve for all neurons

        #Loop through neurons and fit tuning curves
        for j in range(num_nrns): #Neuron number
            if self.encoding_model=='linear':
                tuning=glm_run(y_train,X_b_train[:,j:j+1],input_xy)
            if self.encoding_model=='quadratic':
                tuning=glm_run(y_train_modified,X_b_train[:,j:j+1],input_xy_modified)
            tuning_all[j,:]=np.squeeze(tuning)

        self.tuning_all=tuning_all
        self.input_xy=input_xy

        n=y_train.shape[0]
        dx=np.zeros([n-1,1])
        for i in range(n-1):
            dx[i]=np.sqrt((y_train[i+1,0]-y_train[i,0])**2+(y_train[i+1,1]-y_train[i,1])**2) #Change in state across time steps
        std=np.sqrt(np.mean(dx**2)) # approximate stdev of distribution
        self.std=std 

        #get prior probability of being in each state 
        n_x=np.empty([input_xy.shape[0]])
        for i in range(n):
            loc_idx=np.argmin(cdist(y_train[0:1,:],input_xy))
            n_x[loc_idx]=n_x[loc_idx]+1
        p_x=n_x/n
        self.p_x=p_x

    def predict(self,X_b_test,y_test):

        """
        Predict outcomes using trained tuning curves

        Parameters
        ----------
        X_b_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        y_test: numpy 2d array of shape [n_samples,n_outputs]
            The actual outputs
            This parameter is necesary for the NaiveBayesDecoder  (unlike most other decoders)
            because the first value is nececessary for initialization

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        tuning_all=self.tuning_all
        input_xy=self.input_xy
        std=self.std

        #Get probability of going from one state to the next
        dists = squareform(pdist(input_xy, 'euclidean')) #Distance between all states in "input_xy"
        prob_dists=norm.pdf(dists,0,std)

        #Initializations
        loc_idx= np.argmin(cdist(y_test[0:1,:],input_xy)) #The index of the first location
        num_nrns=tuning_all.shape[0] 
        y_test_predicted=np.empty([X_b_test.shape[0],2]) # matrix of predicted outputs
        num_ts=X_b_test.shape[0] # num time steps we are predicting

        #Loop through time and decode
        for t in range(num_ts):
            ns=X_b_test[t,:] 

            probs_total=np.ones([tuning_all[0,:].shape[0]]) 
            for j in range(num_nrns): #Loop across neurons
                lam=np.copy(tuning_all[j,:]) #Expected spike counts given the tuning curve
                r=ns[j] #Actual spike count
                probs=np.exp(-lam)*lam**r/math.factorial(r) #Probability of the given neuron's spike count given tuning curve (assuming poisson distribution)
                probs_total=np.copy(probs_total*probs) #Update the probability across neurons (probabilities are multiplied across neurons due to the independence assumption)
            prob_dists_vec=np.copy(prob_dists[loc_idx,:]) #Probability of going to all states from the previous state
            
            probs_final=probs_total*prob_dists_vec # final probability (multiply probabilities based on spike count and previous state)
            # probs_final=probs_total*prob_dists_vec*self.p_x # final probability when including p(x), i.e. prior about being in states, which we're not using
            loc_idx=np.argmax(probs_final) #Get the index of the current state (that w/ the highest probability)
            y_test_predicted[t,:]=input_xy[loc_idx,:] 

        return y_test_predicted 
