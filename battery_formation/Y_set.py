import csv
import pandas as pd
import os
import datetime
import time
import numpy as np
import xlrd
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
    if splitLine[-1] == 'xls':
        print(path_list[i])
        data_path = r'/home/lthpc/桌面/battery_formation/battery_formation_data/400mAh/cycling' + '/' + path_list[i]
        dataset = open(data_path, "r", encoding="utf-8-sig")
        data = xlrd.open_workbook(data_path)
        table = data.sheets()[1]
        # load data
        SOH = table.col_values(-1)[1:]
        # the end-of-life threshold
        for j in range(len(SOH)):
            if SOH[j] <= 80:
                ALL_data.append(int(j))
                break
print(ALL_data)
# 250 mAh
path = r'/home/lthpc/桌面/battery_formation/battery_formation_data/250mAh/cycling'
path_list = os.listdir(path)
path_list.sort()
for i in range(len(path_list)):
    splitLine = path_list[i].split('.')
    if splitLine[-1] == 'xls':
        print(path_list[i])
        data_path = r'/home/lthpc/桌面/battery_formation/battery_formation_data/250mAh/cycling' + '/' + path_list[i]
        dataset = open(data_path, "r", encoding="utf-8-sig")
        data = xlrd.open_workbook(data_path)
        table = data.sheets()[1]
        # load data
        SOH = table.col_values(-1)[1:]
        # the end-of-life threshold
        for j in range(len(SOH)):
            if SOH[j] <= 80:
                ALL_data.append(int(j))
                break
# the life of different battery
for i in range(len(ALL_data)):
    with open('TRAIN_Y.csv', 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([ALL_data[i]])
