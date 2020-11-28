import csv
import pandas as pd
import os
import datetime
import time
import numpy as np
# concat the capacity grading data of different batteries
#400 mAh
path = r'/home/lthpc/桌面/battery_formation/battery_formation_data/400mAh/capacity_grading'
path_list = os.listdir(path)
path_list.sort()
input_num = 4
ALL_data = []
num_1 = []
num_2 = []
for i in range(len(path_list)):
    splitLine = path_list[i].split('.')
    if splitLine[-1] == 'csv':
        print(path_list[i])
        data_path = r'/home/lthpc/桌面/battery_formation/battery_formation_data/400mAh/capacity_grading' + '/' + path_list[i]
        csvPD = pd.read_csv(data_path)
        Time = csvPD['Time']
        Discharge_Capacity = csvPD['Discharge Capacity']
        Discharge_Current = csvPD['Discharge Current']
        Discharge_Voltage = csvPD['Discharge Voltage']
        ALL_data.append(Time)
        ALL_data.append(Discharge_Capacity)
        ALL_data.append(Discharge_Current)
        ALL_data.append(Discharge_Voltage)
n_ALL = int(len(ALL_data) / input_num)
print(n_ALL)
#250 mAh
path = r'/home/lthpc/桌面/battery_formation/battery_formation_data/250mAh/capacity_grading'
path_list = os.listdir(path)
path_list.sort()
for i in range(len(path_list)):
    splitLine = path_list[i].split('.')
    if splitLine[-1] == 'csv':
        print(path_list[i])
        data_path = r'/home/lthpc/桌面/battery_formation/battery_formation_data/250mAh/capacity_grading' + '/' + path_list[i]
        csvPD = pd.read_csv(data_path)
        Time = csvPD['Time']
        Discharge_Capacity = csvPD['Discharge Capacity']
        Discharge_Current = csvPD['Discharge Current']
        Discharge_Voltage = csvPD['Discharge Voltage']
        ALL_data.append(Time)
        ALL_data.append(Discharge_Capacity)
        ALL_data.append(Discharge_Current)
        ALL_data.append(Discharge_Voltage)
n_ALL = int(len(ALL_data) / input_num)
print(n_ALL)
#save the capacity grading data of all batteries in train sets
for i in range(len(ALL_data)):
    with open('TRAIN_X_1.csv', 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(ALL_data[i])