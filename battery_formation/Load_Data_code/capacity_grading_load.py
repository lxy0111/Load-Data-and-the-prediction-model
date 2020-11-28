import xlrd
import pandas as pd
import os
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt

data_path = r'/home/lthpc/桌面/battery_formation/battery_formation_data/250mAh/capacity_grading/3-3.xls'
data = xlrd.open_workbook(data_path)
table = data.sheets()[-1]
for i in range(len(table.row_values(0))):
    if table.row_values(0)[i] == '记录序号':
        Number_col = i
    elif table.row_values(0)[i] == '状态':
        State_col = i
    elif table.row_values(0)[i] == '跳转':
        Skip_col = i
    elif table.row_values(0)[i] == '循环':
        Cycle_col = i
    elif table.row_values(0)[i] == '步次':
        Step_col = i
    elif table.row_values(0)[i] == '电流(mA)':
        Current_col = i
    elif table.row_values(0)[i] == '电压(V)':
        Voltage_col = i
    elif table.row_values(0)[i] == '容量(mAh)':
        Capacity_col = i
    elif table.row_values(0)[i] == '能量(mWh)':
        Energy_col = i
    elif table.row_values(0)[i] == '相对时间(h:min:s.ms)':
        Relative_time_col = i
    elif table.row_values(0)[i] == '绝对时间':
        Absolute_time_col = i
# load data
Number = table.col_values(Number_col)[1:]
State = table.col_values(State_col)[1:]
Skip = table.col_values(Skip_col)[1:]
Cycle = table.col_values(Cycle_col)[1:]
Step = table.col_values(Step_col)[1:]
Current = table.col_values(Current_col)[1:]
Voltage = table.col_values(Voltage_col)[1:]
Capacity = table.col_values(Capacity_col)[1:]
Energy = table.col_values(Energy_col)[1:]
Relative_time = table.col_values(Relative_time_col)[1:]
Absolute_time = table.col_values(Absolute_time_col)[1:]
# The charge process
Charge_Start = []
for i in range(1,len(Number)):
    if '充电' in State[i] and Step[i]-Step[i-1] == 1:
        Charge_Start.append(i)
Charge_End = []
if len(Charge_Start) ==0:
    print('Not include charge process')
else:
    for i in range(len(Charge_Start)):
        if len(Charge_Start) == 1:
            if '充电' in State[len(Number) - 1]:
                Charge_End.append(len(Number) - 1)
            else:
                for j in range(Charge_Start[i]+1,len(Number)):
                    if '充电' not in State[j]:
                        Charge_End.append(j - 1)
                        break
        else:
            if i < len(Charge_Start)-1:
                for j in range(Charge_Start[i]+1,Charge_Start[i+1]):
                    if '充电' not in State[j]:
                        Charge_End.append(j-1)
                        break
            else:
                if '充电' in State[len(Number) - 1]:
                    Charge_End.append(len(Number) - 1)
                else:
                    for j in range(Charge_Start[i]+1,len(Number)):
                        if '充电' not in State[j]:
                            Charge_End.append(j - 1)
                            break
# The discharge process
Discharge_Start = []
for i in range(1,len(Number)):
    if '放电' in State[i] and Step[i]-Step[i-1] == 1:
        Discharge_Start.append(i)
Discharge_End = []
if len(Discharge_Start) ==0:
    print('Not include dishcarge process')
else:
    for i in range(len(Discharge_Start)):
        if len(Discharge_Start) == 1:
            if '放电' in State[len(Number)-1]:
                Discharge_End.append(len(Number)-1)
            else:
                for j in range(Discharge_Start[i]+1,len(Number)):
                    if '放电' not in State[j]:
                        Discharge_End.append(j - 1)
                        break

        else:
            if i < len(Discharge_Start)-1:
                for j in range(Discharge_Start[i]+1,Discharge_Start[i+1]):
                    if '放电' not in State[j]:
                        Discharge_End.append(j-1)
                        break
            else:
                if '放电' in State[len(Number) - 1]:
                    Discharge_End.append(len(Number) - 1)
                else:
                    for j in range(Discharge_Start[i]+1,len(Number)):
                        if '放电' not in State[j]:
                            Discharge_End.append(j - 1)
                            break
# the detail data in each charge process
Current_ALL_Charge_process = []
Voltage_ALL_Charge_process = []
Capacity_ALL_Charge_process = []
Energy_ALL_Charge_process = []
Relative_time_ALL_Charge_process = []
Absolute_time_ALL_Charge_process = []
Time_ALL_Charge_process = []
for i in range(len(Charge_Start)):
    if i < len(Charge_Start)-1 :
        Current_Charge_process = Current[Charge_Start[i]:Charge_End[i]+1]
        Voltage_Charge_process = Voltage[Charge_Start[i]:Charge_End[i]+1]
        Capacity_Charge_process = Capacity[Charge_Start[i]:Charge_End[i]+1]
        Energy_Charge_process = Energy[Charge_Start[i]:Charge_End[i]+1]
        Relative_Charge_process = Relative_time[Charge_Start[i]:Charge_End[i]+1]
        Absolute_Charge_process = Absolute_time[Charge_Start[i]:Charge_End[i]+1]
        Time_Charge_process = []
        for j in range(len(Relative_Charge_process)):
            Hour, Min, Sec = Relative_Charge_process[j].split(':')
            time_Charge_process = int(Hour) * 60 + int(Min) + int(Sec[:2]) / 60
            Time_Charge_process.append(time_Charge_process)
        # Append
        Current_ALL_Charge_process.append(Current_Charge_process)
        Voltage_ALL_Charge_process.append(Voltage_Charge_process)
        Capacity_ALL_Charge_process.append(Capacity_Charge_process)
        Energy_ALL_Charge_process.append(Energy_Charge_process)
        Relative_time_ALL_Charge_process.append(Relative_Charge_process)
        Absolute_time_ALL_Charge_process.append(Absolute_Charge_process)
        Time_ALL_Charge_process.append(Time_Charge_process)
    else:
        Current_Charge_process = Current[Charge_Start[i]:Charge_End[i]+1]
        Voltage_Charge_process = Voltage[Charge_Start[i]:Charge_End[i]+1]
        Capacity_Charge_process = Capacity[Charge_Start[i]:Charge_End[i]+1]
        Energy_Charge_process = Energy[Charge_Start[i]:Charge_End[i]+1]
        Relative_Charge_process = Relative_time[Charge_Start[i]:Charge_End[i]+1]
        Absolute_Charge_process = Absolute_time[Charge_Start[i]:Charge_End[i]+1]
        Time_Charge_process = []
        for j in range(len(Relative_Charge_process)):
            Hour, Min, Sec = Relative_Charge_process[j].split(':')
            time_Charge_process = int(Hour) * 60 + int(Min) + int(Sec[:2]) / 60
            Time_Charge_process.append(time_Charge_process)
        # Append
        Current_ALL_Charge_process.append(Current_Charge_process)
        Voltage_ALL_Charge_process.append(Voltage_Charge_process)
        Capacity_ALL_Charge_process.append(Capacity_Charge_process)
        Energy_ALL_Charge_process.append(Energy_Charge_process)
        Relative_time_ALL_Charge_process.append(Relative_Charge_process)
        Absolute_time_ALL_Charge_process.append(Absolute_Charge_process)
        Time_ALL_Charge_process.append(Time_Charge_process)
# the detail data in each discharge process
Current_ALL_Discharge_process = []
Voltage_ALL_Discharge_process = []
Capacity_ALL_Discharge_process = []
Energy_ALL_Discharge_process = []
Relative_time_ALL_Discharge_process = []
Absolute_time_ALL_Discharge_process = []
Time_ALL_Discharge_process = []
for i in range(len(Discharge_End)):
    if i < len(Discharge_End)-1 :
        Current_Discharge_process = Current[Discharge_Start[i]:Discharge_End[i]+1]
        Voltage_Discharge_process = Voltage[Discharge_Start[i]:Discharge_End[i]+1]
        Capacity_Discharge_process = Capacity[Discharge_Start[i]:Discharge_End[i]+1]
        Energy_Discharge_process = Energy[Discharge_Start[i]:Discharge_End[i]+1]
        Relative_Discharge_process = Relative_time[Discharge_Start[i]:Discharge_End[i]+1]
        Absolute_Discharge_process = Absolute_time[Discharge_Start[i]:Discharge_End[i]+1]
        Time_Discharge_process = []
        for j in range(len(Relative_Discharge_process)):
            Hour, Min, Sec = Relative_Discharge_process[j].split(':')
            time_Discharge_process = int(Hour) * 60 + int(Min) + int(Sec[:2]) / 60
            Time_Discharge_process.append(time_Discharge_process)
        # Append
        Current_ALL_Discharge_process.append(Current_Discharge_process)
        Voltage_ALL_Discharge_process.append(Voltage_Discharge_process)
        Capacity_ALL_Discharge_process.append(Capacity_Discharge_process)
        Energy_ALL_Discharge_process.append(Energy_Discharge_process)
        Relative_time_ALL_Discharge_process.append(Relative_Discharge_process)
        Absolute_time_ALL_Discharge_process.append(Absolute_Discharge_process)
        Time_ALL_Discharge_process.append(Time_Discharge_process)
    else:
        Current_Discharge_process = Current[Discharge_Start[i]:Discharge_End[i] + 1]
        Voltage_Discharge_process = Voltage[Discharge_Start[i]:Discharge_End[i] + 1]
        Capacity_Discharge_process = Capacity[Discharge_Start[i]:Discharge_End[i] + 1]
        Energy_Discharge_process = Energy[Discharge_Start[i]:Discharge_End[i] + 1]
        Relative_Discharge_process = Relative_time[Discharge_Start[i]:Discharge_End[i] + 1]
        Absolute_Discharge_process = Absolute_time[Discharge_Start[i]:Discharge_End[i] + 1]
        Time_Discharge_process = []
        for j in range(len(Relative_Discharge_process)):
            Hour, Min, Sec = Relative_Discharge_process[j].split(':')
            time_Discharge_process = int(Hour) * 60 + int(Min) + int(Sec[:2]) / 60
            Time_Discharge_process.append(time_Discharge_process)
        # Append
        Current_ALL_Discharge_process.append(Current_Discharge_process)
        Voltage_ALL_Discharge_process.append(Voltage_Discharge_process)
        Capacity_ALL_Discharge_process.append(Capacity_Discharge_process)
        Energy_ALL_Discharge_process.append(Energy_Discharge_process)
        Relative_time_ALL_Discharge_process.append(Relative_Discharge_process)
        Absolute_time_ALL_Discharge_process.append(Absolute_Discharge_process)
        Time_ALL_Discharge_process.append(Time_Discharge_process)
# Linear interpolation of capacity grading data to the sanme time step of 500.
Time_Steps = len(Time_ALL_Discharge_process[0])
x_new = np.linspace(0,Time_Steps-1,500,endpoint=True)
old_x = []
for i in range(Time_Steps):
    old_x.append(i)
Time_ALL_Discharge_process_new = np.interp(x_new,old_x,Time_ALL_Discharge_process[0])
Capacity_ALL_Discharge_process_new = np.interp(x_new,old_x,Capacity_ALL_Discharge_process[0])
Current_ALL_Discharge_process_new = np.interp(x_new,old_x,Current_ALL_Discharge_process[0])
Voltage_ALL_Discharge_process_new = np.interp(x_new,old_x,Voltage_ALL_Discharge_process[0])
# save the capacity grading data
dataframe = pd.DataFrame({'Time': Time_ALL_Discharge_process_new,
                          'Discharge Capacity':Capacity_ALL_Discharge_process_new,
                          'Discharge Current': Current_ALL_Discharge_process_new,
                          'Discharge Voltage': Voltage_ALL_Discharge_process_new})
dataframe.to_csv("/home/lthpc/桌面/battery_formation/battery_formation_data/250mAh/capacity_grading/3-3-Discharge.csv",index=False,sep=',')
#plot the charge curve
plt.figure(figsize=(3,3),dpi=300)
#Current
ax1 = plt.subplot(221)
for i in range(len(Current_ALL_Charge_process)):
    if i == 0:
        ax1.plot(Time_ALL_Charge_process[i],Current_ALL_Charge_process[i],label=str(i+1)+'st'+' '+'cycle')
    elif i == 1:
        ax1.plot(Time_ALL_Charge_process[i],Current_ALL_Charge_process[i],label=str(i+1)+'nd'+' '+'cycle')
    elif i == 2:
        ax1.plot(Time_ALL_Charge_process[i],Current_ALL_Charge_process[i],label=str(i+1)+'rd'+' '+'cycle')
    else:
        ax1.plot(Time_ALL_Charge_process[i],Current_ALL_Charge_process[i],label=str(i+1)+'th'+' '+'cycle')
ax1.set_title('Charge Current',fontsize=5)
plt.legend(fontsize=2)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.xlabel('Time (min)',fontsize=5)
plt.ylabel('Current (mA)',fontsize=5)
#Voltage
ax2 = plt.subplot(222)
for i in range(len(Voltage_ALL_Charge_process)):
    if i == 0:
        ax2.plot(Time_ALL_Charge_process[i],Voltage_ALL_Charge_process[i],label=str(i+1)+'st'+' '+'cycle')
    elif i == 1:
        ax2.plot(Time_ALL_Charge_process[i],Voltage_ALL_Charge_process[i],label=str(i+1)+'nd'+' '+'cycle')
    elif i == 2:
        ax2.plot(Time_ALL_Charge_process[i],Voltage_ALL_Charge_process[i],label=str(i+1)+'rd'+' '+'cycle')
    else:
        ax2.plot(Time_ALL_Charge_process[i],Voltage_ALL_Charge_process[i],label=str(i+1)+'th'+' '+'cycle')
ax2.set_title('Charge Voltage',fontsize=5)
plt.legend(fontsize=2)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.xlabel('Time (min)',fontsize=5)
plt.ylabel('V (V)',fontsize=5)
# Capacity
ax3 = plt.subplot(223)
for i in range(len(Capacity_ALL_Charge_process)):
    if i == 0:
        ax3.plot(Time_ALL_Charge_process[i],Capacity_ALL_Charge_process[i],label=str(i+1)+'st'+' '+'cycle')
    elif i == 1:
        ax3.plot(Time_ALL_Charge_process[i],Capacity_ALL_Charge_process[i],label=str(i+1)+'nd'+' '+'cycle')
    elif i == 2:
        ax3.plot(Time_ALL_Charge_process[i],Capacity_ALL_Charge_process[i],label=str(i+1)+'rd'+' '+'cycle')
    else:
        ax3.plot(Time_ALL_Charge_process[i],Capacity_ALL_Charge_process[i],label=str(i+1)+'th'+' '+'cycle')
ax3.set_title('Charge Capacity',fontsize=5)
plt.legend(fontsize=2)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.xlabel('Time (min)',fontsize=5)
plt.ylabel('Capacity (mAh)',fontsize=5)
# Energy
ax4 = plt.subplot(224)
for i in range(len(Energy_ALL_Charge_process)):
    if i == 0:
        ax4.plot(Time_ALL_Charge_process[i],Energy_ALL_Charge_process[i],label=str(i+1)+'st'+' '+'cycle')
    elif i == 1:
        ax4.plot(Time_ALL_Charge_process[i],Energy_ALL_Charge_process[i],label=str(i+1)+'nd'+' '+'cycle')
    elif i == 2:
        ax4.plot(Time_ALL_Charge_process[i],Energy_ALL_Charge_process[i],label=str(i+1)+'rd'+' '+'cycle')
    else:
        ax4.plot(Time_ALL_Charge_process[i],Energy_ALL_Charge_process[i],label=str(i+1)+'th'+' '+'cycle')
ax4.set_title('Charge Energy',fontsize=5)
plt.legend(fontsize=2)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.xlabel('Time (min)',fontsize=5)
plt.ylabel('Energy (mWh)',fontsize=5)
plt.tight_layout(pad=1,w_pad=2.0,h_pad=1.0)
plt.suptitle('Charge Process',fontsize=5,x=0.55,y=0.99)
plt.show()

#plot the discharge curve
plt.figure(figsize=(3,3),dpi=300)
#Current
ax1 = plt.subplot(221)
for i in range(len(Current_ALL_Discharge_process)):
    if i == 0:
        ax1.plot(Time_ALL_Discharge_process[i],Current_ALL_Discharge_process[i],label=str(i+1)+'st'+' '+'cycle')
    elif i == 1:
        ax1.plot(Time_ALL_Discharge_process[i],Current_ALL_Discharge_process[i],label=str(i+1)+'nd'+' '+'cycle')
    elif i == 2:
        ax1.plot(Time_ALL_Discharge_process[i],Current_ALL_Discharge_process[i],label=str(i+1)+'rd'+' '+'cycle')
    else:
        ax1.plot(Time_ALL_Discharge_process[i],Current_ALL_Discharge_process[i],label=str(i+1)+'th'+' '+'cycle')
ax1.set_title('Discharge Current',fontsize=5)
plt.legend(fontsize=2)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.xlabel('Time (min)',fontsize=5)
plt.ylabel('Current (mA)',fontsize=5)
#Voltage
ax2 = plt.subplot(222)
for i in range(len(Voltage_ALL_Discharge_process)):
    if i == 0:
        ax2.plot(Time_ALL_Discharge_process[i],Voltage_ALL_Discharge_process[i],label=str(i+1)+'st'+' '+'cycle')
    elif i == 1:
        ax2.plot(Time_ALL_Discharge_process[i],Voltage_ALL_Discharge_process[i],label=str(i+1)+'nd'+' '+'cycle')
    elif i == 2:
        ax2.plot(Time_ALL_Discharge_process[i],Voltage_ALL_Discharge_process[i],label=str(i+1)+'rd'+' '+'cycle')
    else:
        ax2.plot(Time_ALL_Discharge_process[i],Voltage_ALL_Discharge_process[i],label=str(i+1)+'th'+' '+'cycle')
ax2.set_title('Discharge Voltage',fontsize=5)
plt.legend(fontsize=2)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.xlabel('Time (min)',fontsize=5)
plt.ylabel('V (V)',fontsize=5)
# Capacity
ax3 = plt.subplot(223)
for i in range(len(Capacity_ALL_Discharge_process)):
    if i == 0:
        ax3.plot(Time_ALL_Discharge_process[i],Capacity_ALL_Discharge_process[i],label=str(i+1)+'st'+' '+'cycle')
    elif i == 1:
        ax3.plot(Time_ALL_Discharge_process[i],Capacity_ALL_Discharge_process[i],label=str(i+1)+'nd'+' '+'cycle')
    elif i == 2:
        ax3.plot(Time_ALL_Discharge_process[i],Capacity_ALL_Discharge_process[i],label=str(i+1)+'rd'+' '+'cycle')
    else:
        ax3.plot(Time_ALL_Discharge_process[i],Capacity_ALL_Discharge_process[i],label=str(i+1)+'th'+' '+'cycle')
ax3.set_title('Discharge Capacity',fontsize=5)
plt.legend(fontsize=2)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.xlabel('Time (min)',fontsize=5)
plt.ylabel('Capacity (mAh)',fontsize=5)
# Energy
ax4 = plt.subplot(224)
for i in range(len(Energy_ALL_Discharge_process)):
    if i == 0:
        ax4.plot(Time_ALL_Discharge_process[i],Energy_ALL_Discharge_process[i],label=str(i+1)+'st'+' '+'cycle')
    elif i == 1:
        ax4.plot(Time_ALL_Discharge_process[i],Energy_ALL_Discharge_process[i],label=str(i+1)+'nd'+' '+'cycle')
    elif i == 2:
        ax4.plot(Time_ALL_Discharge_process[i],Energy_ALL_Discharge_process[i],label=str(i+1)+'rd'+' '+'cycle')
    else:
        ax4.plot(Time_ALL_Discharge_process[i],Energy_ALL_Discharge_process[i],label=str(i+1)+'th'+' '+'cycle')
ax4.set_title('Discharge Energy', fontsize=5)
plt.legend(fontsize=2)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.xlabel('Time (min)',fontsize=5)
plt.ylabel('Energy (mWh)',fontsize=5)
plt.tight_layout(pad=1,w_pad=2.0,h_pad=1.0)
plt.suptitle('Discharge Process',fontsize=5,x=0.55,y=0.99)
plt.show()





