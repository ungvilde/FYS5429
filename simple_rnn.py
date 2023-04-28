from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout

class SimpleRNNRegression:

    def __init__(self, units=100, num_epochs=10, verbose=1, activation="relu", optimizer="adam"):
         self.units=units
         self.dropout=dropout
         self.num_epochs=num_epochs
         self.verbose=verbose
         self.activation=activation
         self.optimizer=optimizer

    def fit(self, X_train, y_train):

        model=Sequential() 
        model.add(
            SimpleRNN(self.units, input_shape=(X_train.shape[1], X_train.shape[2]),
            recurrent_dropout=self.dropout,
            activation=self.activation)
            ) 
            
        model.add(Dense(y_train.shape[1]))

        model.compile(loss='mse', optimizer=self.optimizer, metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=self.num_epochs, verbose=self.verbose)
        self.model = model


    def predict(self, X_test):
        y_pred = self.model.predict(X_test)

        return y_pred


