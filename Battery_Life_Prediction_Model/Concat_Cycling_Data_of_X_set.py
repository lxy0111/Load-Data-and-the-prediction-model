import csv
import pandas as pd
import os
import datetime
import time
import numpy as np
# concat the first 10 cycles of the cycling data of different batteries
# 400 mAh
path = r'/home/lthpc/桌面/battery_formation/battery_formation_data/400mAh/cycling'
path_list = os.listdir(path)
path_list.sort()
input_num = 8
ALL_data = []
num_1 = []
num_2 = []
for i in range(len(path_list)):
    splitLine = path_list[i].split('.')
    if splitLine[-1] == 'csv':
        print(path_list[i])
        data_path = r'/home/lthpc/桌面/battery_formation/battery_formation_data/400mAh/cycling' + '/' + path_list[i]
        dataset = open(data_path, "r", encoding="utf-8-sig")
        csv_reader_lines = csv.reader(dataset)
        num_ALL_line = 0
        for one_line in csv_reader_lines:
            oneline = []
            for i in one_line:
                one = float(i)
                oneline.append(one)
            n_array = np.array(oneline).reshape(1, -1)
            ALL_data.append(n_array)
            num_ALL_line += 1
            # the first 10 cycles
            if num_ALL_line == 80:
                break
n_ALL = int(len(ALL_data) / input_num)
print(n_ALL)

# 250 mAh
path = r'/home/lthpc/桌面/battery_formation/battery_formation_data/250mAh/cycling'
path_list = os.listdir(path)
path_list.sort()
for i in range(len(path_list)):
    splitLine = path_list[i].split('.')
    if splitLine[-1] == 'csv':
        print(path_list[i])
        data_path = r'/home/lthpc/桌面/battery_formation/battery_formation_data/250mAh/cycling' + '/' + path_list[i]
        dataset = open(data_path, "r", encoding="utf-8-sig")
        csv_reader_lines = csv.reader(dataset)
        num_ALL_line = 0
        for one_line in csv_reader_lines:
            oneline = []
            for i in one_line:
                one = float(i)
                oneline.append(one)
            n_array = np.array(oneline).reshape(1, -1)
            ALL_data.append(n_array)
            num_ALL_line += 1
            # the first 10 cycles 
            if num_ALL_line == 80:
                break
n_ALL = int(len(ALL_data) / input_num)
print(n_ALL)
for i in range(len(ALL_data)):
    with open('TRAIN_X_2.csv', 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(ALL_data[i][0])