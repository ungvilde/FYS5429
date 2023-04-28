from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Activation

class LSTMRegression:
    def __init__(self, units=100, num_epochs=10, optimizer='adam', verbose=1):
         self.units=units
         self.num_epochs=num_epochs
         self.verbose=verbose
         self.optimizer=optimizer

    def fit(self, X_train, y_train):
        model=Sequential() 
        model.add(LSTM(self.units, input_shape=(X_train.shape[1],X_train.shape[2]))) 
        model.add(Dense(y_train.shape[1]))
        model.compile(loss='mse', optimizer=self.optimizer, metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=self.num_epochs, verbose=self.verbose)
        self.model=model


    def predict(self,X_test):
        y_pred = self.model.predict(X_test) #Make predictions
        return y_pred

