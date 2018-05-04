# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 14:01:54 2018

@author: paho

LSTM Network template for regression

Predict real valued output at t+1 given data from t-timesteps...t. Formally, 
LSTM network fits parameters to compute the function:

R^(timesteps, input_dim) --> R^output_dim

Where

timesteps = consecutive timesteps to be considered a single example
input_dim = number of independent variables to be used in regression. Could be
eg. data from various sensors
output_dim = output can be either scalar or vector - define dimensionality here

"""
#%% IMPORTS
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers import Dropout
from keras.utils import Sequence
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
#%% SETTINGS / HYPERPARAMETERS

#The number of entries in each timestep of input, e.g. data from different sensors
input_dim = 6

#The dimension of the predicted variable
output_dim = 1

#number of neurons per lstm layer
LSTM_units = 200

#Additional model / training hyperparameters
dropout_rate = 0.2
test_ratio = 0.05
batch_size = 32
max_epochs = 50

#static normalizing factors for input variables
min_x = np.array(
        [0, 
         60.415733,
         22.111310,
         0.2000000,
         0,
         -39.88021])
max_x = np.array(
        [1792,
         60.520895,
         22.341193,
         57.200000,
         255.99600,
         51.320079])

#Set random seed for reproducible train/test split
random_seed = 42
#%% DATA PREPROCESSING
"""
TODO:
    Järjestä data pituuden mukaan ennen TripSequence() kutsua
    Tee train / test split ennen TripSequence() kutsua
"""
    
class TripBatchGenerator(Sequence):
    """
        A generator class to produce a single batch of sequences 
        for LSTM training
        
        Arguments:
            x_set: The whole training set, array-like of shape (m_examples, 1).
            A single example can be accessed in the manner x_set[example_idx][0]
            and is a numpy array of shape (1, timesteps, n_features). Timesteps
            can vary between examples.
            
            y_set: The labels corresponding to elements in x_set
            
            batch_size: The batch size to be used in training
        
        Outputs:
            batch_x_tensor: Numpy array of shape (batch_size, max_timesteps_batch,
            n_input_features)
            batch_y_tensor: Numpy array of shape (batch_size, max_timesteps_batch,
            n_output_features)
                
    
        #https://keras.io/utils/#sequence
        
    """
    
    
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
    
    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size].copy()
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size].copy()

        #get the max sequence lenght in batch
        max_timesteps_batch = max([seq[0].shape[1] for seq in batch_x])
        
        #initialize return variables as 3D tensors
        batch_x_tensor = np.zeros((len(batch_x), max_timesteps_batch, input_dim))
        batch_y_tensor = np.zeros((len(batch_y), max_timesteps_batch, output_dim))
        
        #Zero pad all samples within batch to max length
        for i in range(len(batch_x)):
            padding_dims = ((0, 0), (0, max_timesteps_batch - batch_x[i][0].shape[1]))
            batch_x[i][0] = np.pad(batch_x[i][0], padding_dims, 'constant', constant_values=(None, 0))
            batch_y[i][0] = np.pad(batch_y[i][0], padding_dims, 'constant', constant_values=(None, 0))
            
            #Reshape to meet Keras expectation
            batch_x[i][0] = np.reshape(batch_x[i][0].transpose(), (1, max_timesteps_batch, input_dim))
            batch_y[i][0] = np.reshape(batch_y[i][0].transpose(), (1, max_timesteps_batch, output_dim))

            #Append x, y to returnable tensor
            batch_x_tensor[i, :, :] = batch_x[i][0]
            batch_y_tensor[i, :, :] = batch_y[i][0]
        
        #Normalize
        #y in range (0, 1) by definition, no need to rescale!
        batch_x_tensor = (np.subtract(batch_x_tensor, min_x)) / (max_x - min_x)

        return batch_x_tensor, batch_y_tensor
    
#Import train data 
data = sio.loadmat('trips_dataset_doubles.mat')
x, y = data['X'], data['Y']

#Get the number of training examples
m_examples = x.shape[0]

#Do a random train/test split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_ratio, random_state = random_seed)


#----------TRAIN-----------
#get the lenghts of the elements in x_train
lengths = np.array([[idx, trip[0].shape[1]] for idx, trip in enumerate(x_train)])
lengths = lengths[lengths[:,1].argsort()]
#order x_train, y_train to ascending order by sequence length
x_train = x_train[lengths[:,0].tolist()]
y_train = y_train[lengths[:,0].tolist()]

#initialize the batch generator to be used in training
ts = TripBatchGenerator(x_train, y_train, batch_size)

#---------TEST-------------
#get the lenghts of the elements in x_test
lengths = np.array([[idx, trip[0].shape[1]] for idx, trip in enumerate(x_test)])
lengths = lengths[lengths[:,1].argsort()]

#order x_train, y_train to ascending order by sequence length
x_test = x_test[lengths[:,0].tolist()]
y_test = y_test[lengths[:,0].tolist()]

#initialize the batch generator to be used in training
ts_test = TripBatchGenerator(x_test, y_test, batch_size)

#%% MODEL

# Initialize the RNN
regressor = Sequential()

#LSTM Layers and dropout regularization
regressor.add(LSTM(units = LSTM_units, return_sequences=True, 
                   input_shape = (None, input_dim)))
regressor.add(Dropout(rate = dropout_rate))

regressor.add(LSTM(units = LSTM_units, return_sequences=True))
regressor.add(Dropout(rate = dropout_rate))

regressor.add(LSTM(units = LSTM_units, return_sequences=True))
regressor.add(Dropout(rate = dropout_rate))

regressor.add(LSTM(units = LSTM_units, return_sequences=True))
regressor.add(Dropout(rate = dropout_rate))

#Linear output layer
regressor.add(TimeDistributed(Dense(1)))

#%% TRAIN

print(regressor.summary())

#Compile and train the model
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit_generator(ts, epochs=max_epochs, verbose=1, validation_data=ts_test)

#%% TEST - Test set

#Get the ground truth
x_testbatch, ground_truth = ts_test.__getitem__(21)

#Make predictions
predicted = regressor.predict(x_testbatch)

#Undo scaling

#Visualize
for i in range(len(predicted)):
    plt.figure()
    plt.plot(ground_truth[i, :], color = 'red', label = 'Ground Truth')
    plt.plot(predicted[i, :], color = 'blue', label = 'Predicted')
    plt.title('Delta SOC')
    plt.xlabel('Timesteps')
    plt.ylabel('SOC %')
    plt.ylim((0.8, 1))
    plt.legend()