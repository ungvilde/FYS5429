# FYS5429
 Neural decoding project for FYS5429. I apply recurrent neural networks and Bayesian methods to spike train data.

# Note
Make sure you have tensorflow installed when running.

# Content
 - **cv_naive_bayes.py** and cv_rnn_lstm.py do 10-fold cross validation for each of the methods
 - **data_preprocessing.py** contain function for data preparation
 - **prepare_data.py** transforms data from raw continuous format to discrete matrix format
 - **lstm.py** has the LSTM decoder
 - **simple_rnn.py** as the simple RNN decoder
 - **naive_bayes.py** has the naive Bayes decoder
 - **metrics.py** has basic metrics
 - **plot_rnn_lstm.py**  is used for plotting
