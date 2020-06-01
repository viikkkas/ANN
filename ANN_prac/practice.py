import pandas as pd
import numpy as np
import tensorflow as tf

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, 13].values

#label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

#one hot encoding for geography
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

#splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#initialize the ANN
ann = tf.keras.models.Sequential()

#add the input layer and first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#add the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#add the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#training the ann
#compiling the ann with an optimizer
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#training the ann on training set
ann.fit(X_train, y_train, batch_size=32, epochs = 100)

#predicting the solution of a single observation
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 6000, 2, 1, 1, 50000]])) > 0.5)

#predicting the test results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
#comparing
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)
