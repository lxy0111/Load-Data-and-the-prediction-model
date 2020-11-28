import warnings
from sklearn import preprocessing
import csv
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D,MaxPool1D,LSTM,BatchNormalization,Conv2D,MaxPool2D
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.models import load_model
from sklearn.svm import SVR
import pickle
import heapq
import h5py
import scipy.io
# set the random seed
np.random.seed(1)
tf.random.set_seed(1)
# Specifying the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Train_X_1 and the input_num = 4
input_num = 4
dataset = open('/home/lthpc/桌面/battery_formation/TRAIN_X_1.csv', "r", encoding="utf-8-sig")
csv_reader_lines = csv.reader(dataset)
ALL_data = []
ALL_data_time_steps = []
num_ALL_line = 0
for one_line in csv_reader_lines:
    while '' in one_line:
        one_line.remove('')
    oneline = []
    for i in one_line:
        one = float(i)
        oneline.append(one)
    n_array = np.array(oneline).reshape(1, -1)
    ALL_time_steps = n_array.shape[1]
    ALL_data_time_steps.append(ALL_time_steps)
    ALL_data.append(n_array)
    num_ALL_line += 1
n_ALL = int(num_ALL_line / input_num)
X_list = []
I_discharge = []
V_discharge = []
Qd_discharge = []
t_discharge = []
Batch_data_time_steps=[]
for i in range(n_ALL):
    if ALL_data_time_steps[i*input_num] == 0:
        continue
    else:
        t_discharge_in_one_cycle = ALL_data[i * input_num][0, :].reshape(1, -1)
        t_discharge.append(t_discharge_in_one_cycle)
        Qd_discharge_in_one_cycle = ALL_data[i * input_num + 1][0, :].reshape(1, -1)
        Qd_discharge.append(Qd_discharge_in_one_cycle)
        I_discharge_min_in_one_cycle = ALL_data[i * input_num + 2][0,:].reshape(1,-1)
        I_discharge.append(I_discharge_min_in_one_cycle)
        V_discharge_in_one_cycle = ALL_data[i * input_num + 3][0, :].reshape(1, -1)
        V_discharge.append(V_discharge_in_one_cycle)
        one_data_time_steps = ALL_data_time_steps[i * input_num]
        Batch_data_time_steps.append(one_data_time_steps)
seq_length = np.array(Batch_data_time_steps)
max_cycle_all_1 = max(Batch_data_time_steps)
x_num_1 = 4
n_ALL = len(Batch_data_time_steps)

for i in range(n_ALL):
    X_list_1 = []
    X_list_1.append(np.array(t_discharge[i]))
    X_list_1.append(np.array(Qd_discharge[i]))
    X_list_1.append(np.array(I_discharge[i]))
    X_list_1.append(np.array(V_discharge[i]))
    _X = np.array(X_list_1).reshape(x_num_1, max_cycle_all_1)
    X_list.append(_X)
X_1 = np.array(X_list)
# Train_X_2 and the input_num = 8
input_num = 8
dataset = open('/home/lthpc/桌面/battery_formation/TRAIN_X_2.csv', "r", encoding="utf-8-sig")
csv_reader_lines = csv.reader(dataset)
ALL_data = []
ALL_data_time_steps = []
num_ALL_line = 0
for one_line in csv_reader_lines:
    while '' in one_line:
        one_line.remove('')
    oneline = []
    for i in one_line:
        one = float(i)
        oneline.append(one)
    n_array = np.array(oneline).reshape(1, -1)
    ALL_time_steps = n_array.shape[1]
    ALL_data_time_steps.append(ALL_time_steps)
    ALL_data.append(n_array)
    num_ALL_line += 1
n_ALL = int(num_ALL_line / input_num)
X_list = []
I_charge = []
I_discharge = []
V_charge = []
V_discharge = []
Qc_charge = []
Qd_discharge = []
t_charge = []
t_discharge = []
Batch_data_time_steps=[]
Batch_dQdV_time_steps=[]
Batch_data_time_steps_discharge=[]
for i in range(n_ALL):
    if ALL_data_time_steps[i*input_num] == 0:
        continue
    else:
        I_charge_min_in_one_cycle = ALL_data[i * input_num ][0,:].reshape(1,-1)
        I_charge.append(I_charge_min_in_one_cycle)
        I_discharge_min_in_one_cycle = ALL_data[i * input_num+1 ][0,:].reshape(1,-1)
        I_discharge.append(I_discharge_min_in_one_cycle)
        V_charge_in_one_cycle = ALL_data[i * input_num + 2][0,:].reshape(1,-1)
        V_charge.append(V_charge_in_one_cycle)
        V_discharge_in_one_cycle = ALL_data[i * input_num + 3][0, :].reshape(1, -1)
        V_discharge.append(V_discharge_in_one_cycle)
        Qc_charge_in_one_cycle = ALL_data[i * input_num + 4][0, :].reshape(1, -1)
        Qc_charge.append(Qc_charge_in_one_cycle)
        Qd_discharge_in_one_cycle = ALL_data[i * input_num + 5][0, :].reshape(1, -1)
        Qd_discharge.append(Qd_discharge_in_one_cycle)
        t_charge_in_one_cycle = ALL_data[i * input_num + 6][0, :].reshape(1, -1)
        t_charge.append(t_charge_in_one_cycle)
        t_discharge_in_one_cycle = ALL_data[i * input_num + 7][0, :].reshape(1, -1)
        t_discharge.append(t_discharge_in_one_cycle)
        one_data_time_steps = ALL_data_time_steps[i * input_num]
        one_data_time_steps_discharge = ALL_data_time_steps[i * input_num+1]
        Batch_data_time_steps.append(one_data_time_steps)
        Batch_data_time_steps_discharge.append(one_data_time_steps_discharge)
seq_length = np.array(Batch_data_time_steps)
max_cycle_all_2 = max(Batch_data_time_steps+Batch_data_time_steps_discharge)
x_num_2 = 8
n_ALL = len(Batch_data_time_steps)
X_list_3 = []
for i in range(int(n_ALL/10)):
    X_list_2 = []
    for j in range(10):
        X_list_1 = []
        X_list_1.append(np.array(I_charge[i*10+j]))
        X_list_1.append(np.array(I_discharge[i*10+j]))
        X_list_1.append(np.array(V_charge[i*10+j]))
        X_list_1.append(np.array(V_discharge[i*10+j]))
        X_list_1.append(np.array(Qc_charge[i*10+j]))
        X_list_1.append(np.array(Qd_discharge[i*10+j]))
        X_list_1.append(np.array(t_charge[i*10+j]))
        X_list_1.append(np.array(t_discharge[i*10+j]))
        _X = np.array(X_list_1).reshape(x_num_2, max_cycle_all_2)
        X_list_2.append(_X)
    X_list_3.append(np.array(X_list_2))
X_2 = np.array(X_list_3)
print(X_2.shape)

# load y
output_num = 1
dataset_Y = open('/home/lthpc/桌面/battery_formation/TRAIN_Y.csv', "r", encoding="utf-8-sig")
csv_reader_lines_Y = csv.reader(dataset_Y)
ALL_data_Y = []
ALL_data_time_steps_Y = []
num_ALL_line_Y = 0
for one_line in csv_reader_lines_Y:
    while '' in one_line:
        one_line.remove('')
    oneline = []
    for i in one_line:
        one = float(i)
        oneline.append(one)
    n_array = np.array(oneline).reshape(1, -1)
    ALL_time_steps = n_array.shape[1]
    ALL_data_time_steps_Y.append(ALL_time_steps)
    ALL_data_Y.append(n_array)
    num_ALL_line_Y += 1
n_ALL_Y = int(num_ALL_line_Y / output_num)
y_num = 1
RCY = []
Y_list = []
Batch_data_time_steps = []
for i in range(n_ALL_Y):
    RCY_in_one_cycle = ALL_data_Y[i * output_num][0,:].reshape(1,-1)
    RCY.append(RCY_in_one_cycle)
Y = np.array(RCY)
# the shape of X_1 should be transposed to (Batch size, time step, input size of train_X_1)
# the shape of X_2 should be transposed to (Batch size, cycle number(10), time step, input size of train_X_2)
X_1 = np.transpose(X_1, (0, 2, 1))
X_2 = np.transpose(X_2, (0, 1, 3,2))
Y = np.transpose(Y, (0, 2, 1))
print(X_1.shape,X_2.shape,Y.shape)
# Cross validation
train_X_1_1 = X_1[0:15]
train_X_1_2 = X_1[18:]
train_X_1 = np.concatenate((train_X_1_1,train_X_1_2),axis=0)

train_X_2_1 = X_2[0:15]
train_X_2_2 = X_2[18:]
train_X_2 = np.concatenate((train_X_2_1,train_X_2_2),axis=0)

train_Y_1 = Y[0:15]
train_Y_2 = Y[18:]
train_Y = np.concatenate((train_Y_1,train_Y_2),axis=0)

test_X_1 = X_1[15:18]
test_X_2 = X_2[15:18]
test_Y = Y[15:18]
# StandardScaler
train_X_1 = train_X_1.reshape(-1, x_num_1)
train_X_2 = train_X_2.reshape(-1, x_num_2)
test_X_1 = test_X_1.reshape(-1, x_num_1)
test_X_2 = test_X_2.reshape(-1, x_num_2)
train_Y = train_Y.reshape(-1, y_num)
test_Y = test_Y.reshape(-1, y_num)

scaler_X_1 = preprocessing.StandardScaler().fit(train_X_1)
train_X_1 = scaler_X_1.transform(train_X_1)
test_X_1 = scaler_X_1.transform(test_X_1)

scaler_X_2 = preprocessing.StandardScaler().fit(train_X_2)
train_X_2 = scaler_X_2.transform(train_X_2)
test_X_2 = scaler_X_2.transform(test_X_2)

scaler_Y = preprocessing.StandardScaler().fit(train_Y)
train_Y = scaler_Y.transform(train_Y)
test_Y = scaler_Y.transform(test_Y)
# the shape of X_1 should be transposed to (Batch size, time step, input size of train_X_1)
# the shape of X_2 should be transposed to (Batch size, cycle number(10), time step, input size of train_X_2)
# the shape of Y should be transposed to (Batch size, output size)

train_X_1 =train_X_1.reshape(-1,max_cycle_all_1,x_num_1)
train_X_2 =train_X_2.reshape(-1,10,max_cycle_all_2,x_num_2)
test_X_1 = test_X_1.reshape(-1,max_cycle_all_1,x_num_1)
test_X_2 = test_X_2.reshape(-1,10,max_cycle_all_2,x_num_2)
train_Y = train_Y.reshape(-1,y_num)
test_Y = test_Y.reshape(-1,y_num)
print(train_X_1.shape,train_X_2.shape,train_Y.shape,test_X_1.shape,test_X_2.shape,test_Y.shape)

time_step_2 = train_X_2.shape[2]

# Restore Model
model = load_model('/home/lthpc/桌面/battery_formation/Model.h5')

pred = model.predict([test_X_1,test_X_2])
rmse = np.mean(pow(np.square(scaler_Y.inverse_transform(test_Y)-scaler_Y.inverse_transform(pred)),0.5))

print('RMSE = ',rmse)
print('True Life = ',np.reshape(scaler_Y.inverse_transform(test_Y),(-1)))
print('Predicted Life = ',np.reshape(scaler_Y.inverse_transform(pred),(-1)))

