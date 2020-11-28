import xlrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_path = r'/home/lthpc/桌面/battery_formation/battery_formation_data/250mAh/formation/1-1.xls'
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
# when the Working Steps i (i=1,2,3) start
Working_Step_Start = []
for i in range(1,len(Number)):
    if State[i] == '恒流充电' and Step[i]-Step[i-1] == 1:
        Working_Step_Start.append(i)
Working_Step_End = []
for i in range(len(Working_Step_Start)):
    if i < len(Working_Step_Start)-1:
        for j in range(Working_Step_Start[i],Working_Step_Start[i+1]):
            if Step[j+1]-Step[j] == 1:
                Working_Step_End.append(j)
    else:
        for j in range(Working_Step_Start[i],len(Number)-1):
            if Step[j+1]-Step[j] == 1:
                Working_Step_End.append(j)
# print(Working_Step_Start)
# the detail data in each Working Step
Current_ALL_Working_Step = []
Voltage_ALL_Working_Step = []
Capacity_ALL_Working_Step = []
Energy_ALL_Working_Step = []
Relative_time_ALL_Working_Step = []
Absolute_time_ALL_Working_Step = []
Time_ALL_Working_Step = []
for i in range(len(Working_Step_Start)):
    if i < len(Working_Step_Start)-1 :
        Current_Working_Step = Current[Working_Step_Start[i]:Working_Step_End[i]+1]
        Voltage_Working_Step = Voltage[Working_Step_Start[i]:Working_Step_End[i]+1]
        Capacity_Working_Step = Capacity[Working_Step_Start[i]:Working_Step_End[i]+1]
        Energy_Working_Step = Energy[Working_Step_Start[i]:Working_Step_End[i]+1]
        Relative_time_Working_Step = Relative_time[Working_Step_Start[i]:Working_Step_End[i]+1]
        Absolute_time_Working_Step = Absolute_time[Working_Step_Start[i]:Working_Step_End[i]+1]
        Time_Working_Step = []
        for j in range(len(Relative_time_Working_Step)):
            Hour, Min, Sec = Relative_time_Working_Step[j].split(':')
            time_Working_Step = int(Hour)*60+int(Min)+int(Sec[:2])/60
            Time_Working_Step.append(time_Working_Step)
        # Append
        Current_ALL_Working_Step.append(Current_Working_Step)
        Voltage_ALL_Working_Step.append(Voltage_Working_Step)
        Capacity_ALL_Working_Step.append(Capacity_Working_Step)
        Energy_ALL_Working_Step.append(Energy_Working_Step)
        Relative_time_ALL_Working_Step.append(Relative_time_Working_Step)
        Absolute_time_ALL_Working_Step.append(Absolute_time_Working_Step)
        Time_ALL_Working_Step.append(Time_Working_Step)
    else:
        Current_Working_Step = Current[Working_Step_Start[i]:Working_Step_End[i]+1]
        Voltage_Working_Step = Voltage[Working_Step_Start[i]:Working_Step_End[i]+1]
        Capacity_Working_Step = Capacity[Working_Step_Start[i]:Working_Step_End[i]+1]
        Energy_Working_Step = Energy[Working_Step_Start[i]:Working_Step_End[i]+1]
        Relative_time_Working_Step = Relative_time[Working_Step_Start[i]:Working_Step_End[i]+1]
        Absolute_time_Working_Step = Absolute_time[Working_Step_Start[i]:Working_Step_End[i]+1]
        Time_Working_Step = []
        for j in range(len(Relative_time_Working_Step)):
            Hour, Min, Sec = Relative_time_Working_Step[j].split(':')
            time_Working_Step = int(Hour)*60+int(Min)+int(Sec[:2])/60
            Time_Working_Step.append(time_Working_Step)
        # Append
        Current_ALL_Working_Step.append(Current_Working_Step)
        Voltage_ALL_Working_Step.append(Voltage_Working_Step)
        Capacity_ALL_Working_Step.append(Capacity_Working_Step)
        Energy_ALL_Working_Step.append(Energy_Working_Step)
        Relative_time_ALL_Working_Step.append(Relative_time_Working_Step)
        Absolute_time_ALL_Working_Step.append(Absolute_time_Working_Step)
        Time_ALL_Working_Step.append(Time_Working_Step)
#plot the curve
plt.figure(figsize=(3,3),dpi=300)
#Current
ax1 = plt.subplot(221)
for i in range(len(Current_ALL_Working_Step)):
    if i == 0:
        ax1.plot(Time_ALL_Working_Step[i],Current_ALL_Working_Step[i],label=str(i+1)+'st'+' '+'cycle')
    elif i == 1:
        ax1.plot(Time_ALL_Working_Step[i],Current_ALL_Working_Step[i],label=str(i+1)+'nd'+' '+'cycle')
    elif i == 2:
        ax1.plot(Time_ALL_Working_Step[i],Current_ALL_Working_Step[i],label=str(i+1)+'rd'+' '+'cycle')
    else:
        ax1.plot(Time_ALL_Working_Step[i],Current_ALL_Working_Step[i],label=str(i+1)+'th'+' '+'cycle')
ax1.set_title('Current Curve',fontsize=5)
plt.legend(fontsize=2)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.xlabel('Time (min)',fontsize=5)
plt.ylabel('Current (mA)',fontsize=5)
#Voltage
ax2 = plt.subplot(222)
for i in range(len(Voltage_ALL_Working_Step)):
    if i == 0:
        ax2.plot(Time_ALL_Working_Step[i],Voltage_ALL_Working_Step[i],label=str(i+1)+'st'+' '+'cycle')
    elif i == 1:
        ax2.plot(Time_ALL_Working_Step[i],Voltage_ALL_Working_Step[i],label=str(i+1)+'nd'+' '+'cycle')
    elif i == 2:
        ax2.plot(Time_ALL_Working_Step[i],Voltage_ALL_Working_Step[i],label=str(i+1)+'rd'+' '+'cycle')
    else:
        ax2.plot(Time_ALL_Working_Step[i],Voltage_ALL_Working_Step[i],label=str(i+1)+'th'+' '+'cycle')
ax2.set_title('Voltage Curve',fontsize=5)
plt.legend(fontsize=2)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.xlabel('Time (min)',fontsize=5)
plt.ylabel('V (V)',fontsize=5)
# Capacity
ax3 = plt.subplot(223)
for i in range(len(Capacity_ALL_Working_Step)):
    if i == 0:
        ax3.plot(Time_ALL_Working_Step[i],Capacity_ALL_Working_Step[i],label=str(i+1)+'st'+' '+'cycle')
    elif i == 1:
        ax3.plot(Time_ALL_Working_Step[i],Capacity_ALL_Working_Step[i],label=str(i+1)+'nd'+' '+'cycle')
    elif i == 2:
        ax3.plot(Time_ALL_Working_Step[i],Capacity_ALL_Working_Step[i],label=str(i+1)+'rd'+' '+'cycle')
    else:
        ax3.plot(Time_ALL_Working_Step[i],Capacity_ALL_Working_Step[i],label=str(i+1)+'th'+' '+'cycle')
ax3.set_title('Capacity Curve',fontsize=5)
plt.legend(fontsize=2)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.xlabel('Time (min)',fontsize=5)
plt.ylabel('Capacity (mAh)',fontsize=5)
# Energy
ax4 = plt.subplot(224)
for i in range(len(Energy_ALL_Working_Step)):
    if i == 0:
        ax4.plot(Time_ALL_Working_Step[i],Energy_ALL_Working_Step[i],label=str(i+1)+'st'+' '+'cycle')
    elif i == 1:
        ax4.plot(Time_ALL_Working_Step[i],Energy_ALL_Working_Step[i],label=str(i+1)+'nd'+' '+'cycle')
    elif i == 2:
        ax4.plot(Time_ALL_Working_Step[i],Energy_ALL_Working_Step[i],label=str(i+1)+'rd'+' '+'cycle')
    else:
        ax4.plot(Time_ALL_Working_Step[i],Energy_ALL_Working_Step[i],label=str(i+1)+'th'+' '+'cycle')
ax4.set_title('Energy Curve',fontsize=5)
plt.legend(fontsize=2)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.xlabel('Time (min)',fontsize=5)
plt.ylabel('Energy (mWh)',fontsize=5)
plt.tight_layout(pad=1,w_pad=2.0,h_pad=1.0)
plt.show()