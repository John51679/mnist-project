import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.optimizers import SGD
from sklearn.model_selection import GridSearchCV
from keras.regularizers import l2
import time

"""Function scale uses the methods of normalization,centering and standardization to reduce big differences between values of the same feature"""
def scale(data_input,mode="Normalize"):
    cols = len(data_input[0,:])
    data_input = np.float64(data_input)
    if mode == "Normalize":
        for i in range(cols):
            m = max(data_input[:,i])
            if m != 0: data_input[:,i] = np.divide(data_input[:,i],m)
    elif mode == "Standardize":
        for i in range(cols):
            if data_input[:, i].std() != 0:
                data_input[:, i] = np.divide(np.subtract(data_input[:, i], data_input[:, i].mean()),data_input[:, i].std())
            else:
                data_input[:, i] = np.subtract(data_input[:, i], data_input[:, i].mean())
    elif mode == "Centering":
        for i in range(cols):
            data_input[:, i] = np.subtract(data_input[:, i], data_input[:, i].mean())
    else:
        print("The mode you have entered is invalid. Try using 'Normalize' or 'Standardize' or 'Centering'")
        exit(1)
    return data_input

"""Model function is going to be used by GridSearchCV to create our ANN model"""
def model(learn_rate,momentum,L2):
    model = Sequential()
    model.add(Dense(397, input_dim=784, activation='relu',kernel_regularizer=l2(L2)))
    model.add(Dense(198,activation='relu',kernel_regularizer=l2(L2)))
    model.add(Dense(10, activation='sigmoid'))
    optimizer = SGD(learning_rate=learn_rate,momentum=momentum)
    """we have two model.compile below. The first uses CE loss function and the other uses MSE"""
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    #model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    return model

start = time.time() #begin timer

"""Pre-processing"""
trainDF = pd.read_csv("mnist_train.csv")
testDF = pd.read_csv("mnist_test.csv")

train_input = trainDF.iloc[:,1:].values
train_output = trainDF.iloc[:,0].values
test_input = testDF.iloc[:,1:].values
test_output = testDF.iloc[:,0].values

train_input = scale(train_input,'Standardize')
test_input = scale(test_input,'Standardize')

print("palegkas gay")

trainEN = np_utils.to_categorical(train_output).astype(float) #Encode training output to vectors of 0 and 1
testEN = np_utils.to_categorical(test_output).astype(float) #Encode testing output to vectors of 0 and 1

"""end of pre-processing"""

"""Creating our architecture"""

"""Here kerasClassifier is a class that belongs to scikit_learn. We use it to link GridSearchCV with Keras Sequential model"""
estimator = KerasClassifier(build_fn=model, epochs=100, batch_size=100, verbose = 0)
lr = [0.001] #Learn rate. Can be modified from here.
momentum = [0.6] #momentum. Can be modified from here.
L2 = [0.0] #L2 regularizer. Can be modified from here.
param_grid = dict(learn_rate = lr,momentum = momentum, L2 = L2) #The hyperparameters that will be passed into our model

"""include 5 - fold cross validation using GridSearchCV. Here n_jobs defines whether parallelism will be used or not. the value of -1 means that all CPU cores will be used"""
grid = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-1, cv=5)
grid_result = grid.fit(train_input, trainEN) #Begin training
ANN_model = grid_result.best_estimator_.model #We save the model here.

json_f = ANN_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(json_f)
ANN_model.save_weights("model.h5")


"""End of model"""

"""Plot accuracy - loss"""
history = ANN_model.history

plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('My model')
plt.ylabel('Accuracy - Loss')
plt.xlabel('epoch')
plt.legend(['accuracy', 'loss'], loc='upper left')
plt.show()

"""End of plotting"""

"""Evaluate model in a completely new dataset"""
_,Accuracy = ANN_model.evaluate(test_input,testEN)
print("Accuracy of test set is " , Accuracy*100 , "%" )

"""End of Evaluation"""

"""Print results for training accuracy and hyperparameters used. Also display the timer."""
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (Std = %f) with: %r" % (mean, stdev, param))

end = time.time()
print("Total time estimated: {0}".format(end-start))

"""End"""