import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
    

training_data = pd.read_csv("dataset.csv")
    
X = training_data.iloc[ : , :-1].values
y = training_data.iloc[:, -1:].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)      

#Initializing Neural Network
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(21, activation = 'relu', input_dim = 30))

# Adding the output layer
classifier.add(Dense(1, activation = 'sigmoid'))

# Compiling Neural Network
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting our model 
classifier.fit(X_train, y_train, batch_size = 9, nb_epoch = 10)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
y_test = (y_test > 0)
# Creating the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print ("\n\n")
print (cm)
print("\n")
accuracy = 100.0 * accuracy_score(y_test, y_pred)
print ("The accuracy on testing data is: ", str(accuracy))
