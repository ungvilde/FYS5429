import numpy as np
from scipy import io
from data_preprocessing import *
import pickle


for datafile in ['hc', 'm1', 's1']:
    data=io.loadmat('data/'+ datafile + '_data_raw.mat')
    spike_times=data['spike_times'] 
    dt=.05 #Size of time bins (in seconds)

    # spike_times=data['spike_times'] #Load spike times of all neurons
    if datafile in ['m1', 's1']:
        vels=data['vels'] #Load x and y velocities
        vel_times=data['vel_times'] #Load times at which velocities were recorded

        t_start=vel_times[0] #Time to start extracting data - here the first time velocity was recorded
        t_end=vel_times[-1]
        downsample_factor=1

        #When loading the Matlab cell "spike_times", Python puts it in a format with an extra unnecessary dimension
        #First, we will put spike_times in a cleaner format: an array of arrays
        spike_times=np.squeeze(spike_times)
        for i in range(spike_times.shape[0]):
            spike_times[i]=np.squeeze(spike_times[i])

        #Bin output (velocity) data using "bin_output" function
        vels_binned=bin_output(vels, vel_times, dt, t_start, t_end, downsample_factor)

        #### FORMAT INPUT ####
        #Bin neural data using "bin_spikes" function
        neural_data=bin_spikes(spike_times, dt, t_start, t_end)

        with open('data/'+datafile+'_neural_data.pickle','wb') as f:
            pickle.dump([neural_data, vels_binned],f)
    else:
        pos=data['pos']
        pos_times=data['pos_times'][0] 
        dt=.2

        t_start=pos_times[0] #Time to start extracting data - here the first time velocity was recorded
        t_end=5608
        downsample_factor=1

        #When loading the Matlab cell "spike_times", Python puts it in a format with an extra unnecessary dimension
        #First, we will put spike_times in a cleaner format: an array of arrays
        spike_times=np.squeeze(spike_times)
        for i in range(spike_times.shape[0]):
            spike_times[i]=np.squeeze(spike_times[i])

        #Bin output (position) data using "bin_output" function
        pos_binned=bin_output(pos, pos_times, dt, t_start, t_end, downsample_factor)

        #Bin neural data using "bin_spikes" function
        neural_data=bin_spikes(spike_times, dt, t_start, t_end)

        data_folder='data/' 

        with open('data/'+datafile+'_neural_data.pickle','wb') as f:
            pickle.dump([neural_data, pos_binned], f)