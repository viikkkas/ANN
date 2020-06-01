import numpy as np
import pandas as pd
import tensorflow as tf

#importing dataset
dataset = pd.read_excel('Folds5x2_pp.xlsx')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

#splitting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#initializing the ann
ann = tf.keras.models.Sequential()

#adding layers
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 1)) #no output activation function for regression
#sigmoid is used in case of classification

#compiling the ann with optimizer and loss function
#optimizer is used for stochastic gradient descent
ann.compile(optimizer = 'adam', loss = 'mean_squared_error')

#training the model
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)
#forward propagation takes place in batches

#predicting the results
y_pred = ann.predict(X_test)
np.set_printoptions(precision = 2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))
