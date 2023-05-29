import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error as mse
import seaborn as sns
import itertools 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time

import os 
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow import keras
import random
seed_value=0
random.seed(seed_value)
os.environ['PYTHONHASHSEED']=str(seed_value)

np.random.seed(seed_value)

tf.random.set_seed(seed_value)


session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

def ModelLSTM(TimeDimension,FeaturesDimension, LSTMoutputDimension,OutputDimension, return_seq=False):
    
    input_ = keras.layers.Input(shape=(TimeDimension, FeaturesDimension))
    x = keras.layers.LSTM(LSTMoutputDimension,return_sequences=return_seq)(input_)
    if return_seq==True:
        x= tf.keras.layers.Flatten()(x)
    x=keras.layers.Dense(int(LSTMoutputDimension*TimeDimension/2),activation = 'relu')(x)
    x=keras.layers.Dense(int(LSTMoutputDimension*TimeDimension/4),activation = 'relu')(x)
    
    output = keras.layers.Dense(OutputDimension)(x)
    model_LSTM = keras.models.Model(inputs=input_, outputs=output)
    return model_LSTM
	
path_list = os.listdir('Dataset')

#print experiments
cumulative_inputs=[]
cumulative_targets=[]
n_past = 10
for exp,path in enumerate(path_list[:-1]):
    trajectory = np.load(path)
    trajectory = trajectory[:200]
    #print(trajectory)
    inputs_tmp = np.lib.stride_tricks.sliding_window_view(trajectory,(n_past+1,2))
    targets_nn = []
    inputs_nn=[]
    for i in range(len(inputs_tmp)-1):
        inputs_nn.append(inputs_tmp[i][0])
        targets_nn.append(inputs_tmp[i+1][0][-1][0])
        
    fig,ax=plt.subplots(figsize=(10,5))
    plt.plot(np.arange(0,len(targets_nn))*15,targets_nn,linewidth = 3, c = '#8FDB6E')
    plt.grid(linestyle = 'dashed',alpha = 0.7)
    plt.title(f'{path}')
    plt.show()
    
    u=[]
    for el in inputs_nn:
        u.append(el[-1][1])
    fig,ax=plt.subplots(figsize=(10,3))
    plt.step(np.arange(0,len(targets_nn))*15,u[-len(targets_nn):],linewidth = 3, c =  '#F16996')
    plt.yticks(ticks=[0,1],labels = ['NO TETRA','TETRA'],fontsize = 10,horizontalalignment='center',rotation = 90,rotation_mode = 'anchor')
    #plt.ylabel('Input')

    plt.grid(linestyle = 'dashed',alpha = 0.7)
    plt.xlabel('Time [min]')
    plt.show()
    print()

    
    if exp ==0:
        cumulative_inputs = inputs_nn
        cumulative_targets = targets_nn
    else:
        cumulative_inputs=np.vstack([cumulative_inputs,inputs_nn])
        cumulative_targets=np.hstack([cumulative_targets,targets_nn])
        

ne=200
FeaturesDimension = 2
TimeDimension = n_past+1 #timesteps
LSTMoutputDimension = 10
OutputDimension = 1
lr=0.01

callbacks=EarlyStopping(monitor='loss', patience=100,min_delta=0.000001)
tf.keras.backend.clear_session()
keras.initializers.RandomUniform(minval = -0.05, maxval = 0.05, seed = None)

model_LSTM=ModelLSTM(TimeDimension,FeaturesDimension, LSTMoutputDimension,OutputDimension,return_seq=True)

opt = keras.optimizers.Adam(learning_rate=lr)
model_LSTM.compile(optimizer = opt, loss = 'mean_squared_error')
history=model_LSTM.fit(tf.convert_to_tensor(cumulative_inputs),tf.convert_to_tensor(cumulative_targets),batch_size=len(targets_nn),epochs=ne,verbose=0,callbacks=callbacks,validation_split = 0.2)
plt.plot((history.history['loss']),label = 'loss') #anche 246
plt.plot((history.history['val_loss']))
#plt.yscale('log')
plt.legend()
plt.show()

path = path_list[-1]
trajectory = np.load(path)
trajectory = trajectory[30:200]
inputs_tmp = np.lib.stride_tricks.sliding_window_view(trajectory,(n_past+1,2))
targets_nn = []
inputs_nn=[]
for i in range(len(inputs_tmp)-1):
    inputs_nn.append(inputs_tmp[i][0])
    targets_nn.append(inputs_tmp[i+1][0][-1][0])

fig,ax=plt.subplots(figsize=(10,5))
plt.plot(np.arange(0,len(targets_nn))*15,targets_nn,linewidth = 3, c = '#8FDB6E')
plt.grid(linestyle = 'dashed',alpha = 0.7)
plt.title(f'{path}')
plt.show()

 
u=[]
for el in inputs_nn:
    u.append(el[-1][1])
fig,ax=plt.subplots(figsize=(10,3))
plt.step(np.arange(0,len(targets_nn))*15,u[-len(targets_nn):],linewidth = 3, c =  '#F16996')
plt.yticks(ticks=[0,1],labels = ['NO TETRA','TETRA'],fontsize = 10,horizontalalignment='center',rotation = 90,rotation_mode = 'anchor')
#plt.ylabel('Input')

plt.grid(linestyle = 'dashed',alpha = 0.7)
plt.xlabel('Time [min]')
plt.show()
print()

predictions = model_LSTM.predict(tf.convert_to_tensor(inputs_nn))
 
fig,ax=plt.subplots(figsize=(10,5))

plt.plot(np.arange(0,len(targets_nn))*15,targets_nn,label='True',linewidth = 3,c = '#8FDB6E')
plt.plot(np.arange(0,len(targets_nn))*15,predictions,label='Pred',linewidth = 3,alpha = 0.7, linestyle = 'dashed')
plt.grid(linestyle = 'dashed',alpha = 0.7)
plt.title(f'{path}')
plt.legend()
plt.show()

u=[]
for el in inputs_nn:
    u.append(el[-1][1])
fig,ax=plt.subplots(figsize=(10,3))
plt.step(np.arange(0,len(predictions))*15,u[-len(predictions):],linewidth = 3, c =  '#F16996')
plt.yticks(ticks=[0,1],labels = ['NO TETRA','TETRA'],fontsize = 10,horizontalalignment='center',rotation = 90,rotation_mode = 'anchor')
#plt.ylabel('Input')

plt.grid(linestyle = 'dashed',alpha = 0.7)
plt.xlabel('Time [min]')
plt.show()

cumulative_inputs=[]
cumulative_targets=[]

for exp,path in enumerate(path_list):
    trajectory = np.load(path)
    if '111' in path:
        i = 30
    else:
        i = 0
    trajectory = trajectory[i:200]
    inputs_tmp = np.lib.stride_tricks.sliding_window_view(trajectory,(n_past+1,2))
    targets_nn = []
    inputs_nn=[]
    for i in range(len(inputs_tmp)-1):
        inputs_nn.append(inputs_tmp[i][0])
        targets_nn.append(inputs_tmp[i+1][0][-1][0])
        
    
    if exp ==0:
        cumulative_inputs = inputs_nn
        cumulative_targets = targets_nn
    else:
        cumulative_inputs=np.vstack([cumulative_inputs,inputs_nn])
        cumulative_targets=np.hstack([cumulative_targets,targets_nn])
        



callbacks=EarlyStopping(monitor='loss', patience=100,min_delta=0.000001)
tf.keras.backend.clear_session()
keras.initializers.RandomUniform(minval = -0.05, maxval = 0.05, seed = None)

model_LSTM=ModelLSTM(TimeDimension,FeaturesDimension, LSTMoutputDimension,OutputDimension,return_seq=True)

opt = keras.optimizers.Adam(learning_rate=lr)
model_LSTM.compile(optimizer = opt, loss = 'mean_squared_error')
history=model_LSTM.fit(tf.convert_to_tensor(cumulative_inputs),tf.convert_to_tensor(cumulative_targets),batch_size=len(targets_nn),epochs=ne,verbose=0,callbacks=callbacks)
plt.plot((history.history['loss'])) #anche 246
plt.show()

model_LSTM.save('myLSTM_RealWorld')
