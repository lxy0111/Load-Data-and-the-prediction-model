import xlrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 循环数据读取
data_path = r'C:/Users/黄耀迪/Downloads/69#.xls'
data = xlrd.open_workbook(data_path)
table_name = data.sheet_names()
Detail_Sheet_Index = []
for i in range(len(table_name)):
    if 'Detail' in table_name[i]:
        Detail_Sheet_Index.append(i)
Current_ALL_Charge_process = []
Voltage_ALL_Charge_process = []
Capacity_ALL_Charge_process = []
Energy_ALL_Charge_process = []
Relative_time_ALL_Charge_process = []
Absolute_time_ALL_Charge_process = []
Current_ALL_Discharge_process = []
Voltage_ALL_Discharge_process = []
Capacity_ALL_Discharge_process = []
Energy_ALL_Discharge_process = []
Relative_time_ALL_Discharge_process = []
Absolute_time_ALL_Discharge_process = []
Cycle_end  = None
for i in range(len(Detail_Sheet_Index)):
    if i == 0:
        table = data.sheets()[Detail_Sheet_Index[i]]
        for j in range(len(table.row_values(0))):
            if table.row_values(0)[j] == '记录序号':
                Number_col = j
            elif table.row_values(0)[j] == '状态':
                State_col = j
            elif table.row_values(0)[j] == '跳转':
                Skip_col = j
            elif table.row_values(0)[j] == '循环':
                Cycle_col = j
            elif table.row_values(0)[j] == '步次':
                Step_col = j
            elif table.row_values(0)[j] == '电流(mA)':
                Current_col = j
            elif table.row_values(0)[j] == '电压(V)':
                Voltage_col = j
            elif table.row_values(0)[j] == '容量(mAh)':
                Capacity_col = j
            elif table.row_values(0)[j] == '能量(mWh)':
                Energy_col = j
            elif table.row_values(0)[j] == '相对时间(h:min:s.ms)':
                Relative_time_col = j
            elif table.row_values(0)[j] == '绝对时间':
                Absolute_time_col = j
        # load data
        Number = np.array(table.col_values(Number_col)[1:])
        State = np.array(table.col_values(State_col)[1:])
        Skip = np.array(table.col_values(Skip_col)[1:])
        Cycle = np.array(table.col_values(Cycle_col)[1:])
        Step = np.array(table.col_values(Step_col)[1:])
        Current = np.array(table.col_values(Current_col)[1:])
        Voltage = np.array(table.col_values(Voltage_col)[1:])
        Capacity = np.array(table.col_values(Capacity_col)[1:])
        Energy = np.array(table.col_values(Energy_col)[1:])
        Relative_time = np.array(table.col_values(Relative_time_col)[1:])
        Absolute_time = np.array(table.col_values(Absolute_time_col)[1:])
    else:
        table = data.sheets()[Detail_Sheet_Index[i]]
        for j in range(len(table.row_values(0))):
            if table.row_values(0)[j] == '记录序号':
                Number_col = j
            elif table.row_values(0)[j] == '状态':
                State_col = j
            elif table.row_values(0)[j] == '跳转':
                Skip_col = j
            elif table.row_values(0)[j] == '循环':
                Cycle_col = j
            elif table.row_values(0)[j] == '步次':
                Step_col = j
            elif table.row_values(0)[j] == '电流(mA)':
                Current_col = j
            elif table.row_values(0)[j] == '电压(V)':
                Voltage_col = j
            elif table.row_values(0)[j] == '容量(mAh)':
                Capacity_col = j
            elif table.row_values(0)[j] == '能量(mWh)':
                Energy_col = j
            elif table.row_values(0)[j] == '相对时间(h:min:s.ms)':
                Relative_time_col = j
            elif table.row_values(0)[j] == '绝对时间':
                Absolute_time_col = j
        # load data
        Number = np.concatenate([Number,np.array(table.col_values(Number_col)[1:])],axis=0)
        State = np.concatenate([State,np.array(table.col_values(State_col)[1:])],axis=0)
        Skip = np.concatenate([Skip,np.array(table.col_values(Skip_col)[1:])],axis=0)
        Cycle = np.concatenate([Cycle,np.array(table.col_values(Cycle_col)[1:])],axis=0)
        Step = np.concatenate([Step,np.array(table.col_values(Step_col)[1:])],axis=0)
        Current = np.concatenate([Current,np.array(table.col_values(Current_col)[1:])],axis=0)
        Voltage = np.concatenate([Voltage,np.array(table.col_values(Voltage_col)[1:])],axis=0)
        Capacity = np.concatenate([Capacity,np.array(table.col_values(Capacity_col)[1:])],axis=0)
        Energy = np.concatenate([Energy,np.array(table.col_values(Energy_col)[1:])],axis=0)
        Relative_time = np.concatenate([Relative_time,np.array(table.col_values(Relative_time_col)[1:])],axis=0)
        Absolute_time = np.concatenate([Relative_time,np.array(table.col_values(Absolute_time_col)[1:])],axis=0)
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

#plot the charge curve
fig = plt.figure(figsize=(3,3),dpi=300)
#Current
grid = plt.GridSpec(15,9)
camp = plt.get_cmap('Blues')
ax1 = fig.add_subplot(grid[:4,:4])
ax1.set_prop_cycle(color=[camp(1*i/len(Current_ALL_Charge_process)) for i in range(len(Current_ALL_Charge_process))])
for i in range(len(Current_ALL_Charge_process)):
    ax1.plot(Time_ALL_Charge_process[i],Current_ALL_Charge_process[i])
ax1.set_title('Charge Current',fontsize=3)
plt.xticks(fontsize=3)
plt.yticks(fontsize=3)
plt.xlabel('Time (min)',fontsize=3)
plt.ylabel('Current (mA)',fontsize=3)
#Voltage
ax2 = fig.add_subplot(grid[:4,5:9])
ax2.set_prop_cycle(color=[camp(1*i/len(Voltage_ALL_Charge_process)) for i in range(len(Voltage_ALL_Charge_process))])
for i in range(len(Voltage_ALL_Charge_process)):
    ax2.plot(Time_ALL_Charge_process[i],Voltage_ALL_Charge_process[i])
ax2.set_title('Charge Voltage',fontsize=3)
plt.xticks(fontsize=3)
plt.yticks(fontsize=3)
plt.xlabel('Time (min)',fontsize=3)
plt.ylabel('V (V)',fontsize=3)
# Capacity
ax3 = fig.add_subplot(grid[7:11,:4])
ax3.set_prop_cycle(color=[camp(1*i/len(Time_ALL_Charge_process)) for i in range(len(Time_ALL_Charge_process))])
for i in range(len(Capacity_ALL_Charge_process)):
    ax3.plot(Time_ALL_Charge_process[i],Capacity_ALL_Charge_process[i])
plt.xticks(fontsize=3)
plt.yticks(fontsize=3)
ax3.set_title('Charge Capacity',fontsize=3)
plt.xlabel('Time (min)',fontsize=3)
plt.ylabel('Capacity (mAh)',fontsize=3)
# Energy
ax4 = fig.add_subplot(grid[7:11,5:9])
ax4.set_prop_cycle(color=[camp(1*i/len(Energy_ALL_Charge_process)) for i in range(len(Energy_ALL_Charge_process))])
for i in range(len(Energy_ALL_Charge_process)):
    ax4.plot(Time_ALL_Charge_process[i],Energy_ALL_Charge_process[i])
ax4.set_title('Charge Energy',fontsize=3)
plt.xticks(fontsize=3)
plt.yticks(fontsize=3)
plt.xlabel('Time (min)',fontsize=3)
plt.ylabel('Energy (mWh)',fontsize=3)
#color bar
ax5 = fig.add_subplot(grid[14,:])
ax5.set_prop_cycle(color=[camp(1*i/len(Energy_ALL_Charge_process)) for i in range(len(Energy_ALL_Charge_process))])
cycle_number = np.ones([len(Energy_ALL_Charge_process),1])
for i in range(len(Energy_ALL_Charge_process)):
    ax5.barh(np.array([0]),cycle_number[i],left=np.sum(cycle_number[:i],axis=0))
ax5.set_title('Color Bar',fontsize=3)
ax5.spines['left'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.spines['top'].set_visible(False)
ax5.spines['bottom'].set_visible(False)
plt.yticks([])
plt.xticks(fontsize=3)
plt.xlabel('Cycles',fontsize=3)
plt.suptitle('Charge Process',fontsize=5,x=0.5,y=0.95)
plt.show()

#plot the discharge curve
fig = plt.figure(figsize=(3,3),dpi=300)
#Current
grid = plt.GridSpec(15,9)
camp = plt.get_cmap('Blues')
ax1 = fig.add_subplot(grid[:4,:4])
ax1.set_prop_cycle(color=[camp(1*i/len(Current_ALL_Discharge_process)) for i in range(len(Current_ALL_Discharge_process))])
for i in range(len(Current_ALL_Discharge_process)):
    ax1.plot(Time_ALL_Discharge_process[i],Current_ALL_Discharge_process[i])
ax1.set_title('Discharge Current',fontsize=3)
plt.xticks(fontsize=3)
plt.yticks(fontsize=3)
plt.xlabel('Time (min)',fontsize=3)
plt.ylabel('Current (mA)',fontsize=3)
#Voltage
ax2 = fig.add_subplot(grid[:4,5:9])
ax2.set_prop_cycle(color=[camp(1*i/len(Voltage_ALL_Discharge_process)) for i in range(len(Voltage_ALL_Discharge_process))])
for i in range(len(Voltage_ALL_Discharge_process)):
    ax2.plot(Time_ALL_Discharge_process[i],Voltage_ALL_Discharge_process[i])
ax2.set_title('Discharge Voltage',fontsize=3)
plt.xticks(fontsize=3)
plt.yticks(fontsize=3)
plt.xlabel('Time (min)',fontsize=3)
plt.ylabel('V (V)',fontsize=3)
# Capacity
ax3 = fig.add_subplot(grid[7:11,:4])
ax3.set_prop_cycle(color=[camp(1*i/len(Capacity_ALL_Discharge_process)) for i in range(len(Capacity_ALL_Discharge_process))])
for i in range(len(Capacity_ALL_Discharge_process)):
    ax3.plot(Time_ALL_Discharge_process[i],Capacity_ALL_Discharge_process[i])
ax3.set_title('Discharge Capacity',fontsize=3)
plt.xticks(fontsize=3)
plt.yticks(fontsize=3)
plt.xlabel('Time (min)',fontsize=3)
plt.ylabel('Capacity (mAh)',fontsize=3)
# Energy
ax4 = fig.add_subplot(grid[7:11,5:9])
ax4.set_prop_cycle(color=[camp(1*i/len(Energy_ALL_Discharge_process)) for i in range(len(Energy_ALL_Discharge_process))])
for i in range(len(Energy_ALL_Discharge_process)):
    ax4.plot(Time_ALL_Discharge_process[i],Energy_ALL_Discharge_process[i])
ax4.set_title('Discharge Energy',fontsize=3)
plt.xticks(fontsize=3)
plt.yticks(fontsize=3)
plt.xlabel('Time (min)',fontsize=3)
plt.ylabel('Energy (mWh)',fontsize=3)
#color bar
ax5 = fig.add_subplot(grid[14,:])
ax5.set_prop_cycle(color=[camp(1*i/len(Energy_ALL_Discharge_process)) for i in range(len(Energy_ALL_Discharge_process))])
cycle_number = np.ones([len(Energy_ALL_Discharge_process),1])
for i in range(len(Energy_ALL_Discharge_process)):
    ax5.barh(np.array([0]),cycle_number[i],left=np.sum(cycle_number[:i],axis=0))
ax5.set_title('Color Bar',fontsize=3)
ax5.spines['left'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.spines['top'].set_visible(False)
ax5.spines['bottom'].set_visible(False)
plt.yticks([])
plt.xticks(fontsize=3)
plt.xlabel('Cycles',fontsize=3)
plt.suptitle('Discharge Process',fontsize=5,x=0.5,y=0.95)
plt.show()

#Charge/Discharge Capacity during whole life
Charge_Cap = []
Discharge_Cap = []
for i in range(len(Capacity_ALL_Charge_process)):
    Charge_Cap.append(Capacity_ALL_Charge_process[i][-1])
for i in range(len(Capacity_ALL_Discharge_process)):
    Discharge_Cap.append(Capacity_ALL_Discharge_process[i][-1])
fig = plt.figure(figsize=(3,3),dpi=300)
grid = plt.GridSpec(5,5)
Cycle_Charge = [i for i in range(1,len(Charge_Cap)+1)]
Cycle_Discharge = [i for i in range(1,len(Discharge_Cap)+1)]
#charge capacity
ax1 = fig.add_subplot(grid[:2,:])
cc = ax1.scatter(Cycle_Charge,Charge_Cap,c=Cycle_Charge,cmap='Blues',s=3)
ax1.set_title('Charge Capacity',fontsize=3)
plt.xticks(fontsize=3)
plt.yticks(fontsize=3)
plt.xlabel('Cycle Number',fontsize=3)
plt.ylabel('Capacity (mAh)',fontsize=3)
#discharge capacity
ax2 = fig.add_subplot(grid[3:,:])
dc = ax2.scatter(Cycle_Discharge,Discharge_Cap,c=Cycle_Discharge,cmap='Blues',s=3)
ax2.set_title('Discharge Capacity',fontsize=3)
plt.xticks(fontsize=3)
plt.yticks(fontsize=3)
plt.xlabel('Cycle Number',fontsize=3)
plt.ylabel('Capacity (mAh)',fontsize=3)
cb1 = fig.colorbar(cc,ax=ax1)
cb2 = fig.colorbar(dc,ax=ax2)
cb1.ax.tick_params(labelsize=3)
cb2.ax.tick_params(labelsize=3)
plt.show()











