import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
# plt.figure(figsize=(10,6))
raw_data_read = pd.read_csv('D:\Marketing Raw Data.csv')

# print(raw_data_read.head())

# print(raw_data_read['Visitors'])
# plt.plot(raw_data_read['Visitors'])
# plt.xticks(range(0,len(raw_data_read['Visitors'])),raw_data_read['Date'],rotation=90)
# plt.show()


a = raw_data_read.tail(10).reset_index()

# plt.plot(a['Visitors'],label='Number of Visitors',linewidth=3)
# plt.plot(a['Revenue'],label='Revenue',linestyle='--',linewidth=3)
# plt.title('RAW DATA TO THIS',fontweight='bold')

# plt.xticks(range(0,len(a['Visitors'])),a['Date'],rotation=90)
# plt.legend(loc='upper right',fontsize=8)
# plt.xlabel('Date')
# plt.show()

# bar

# plt.bar(range(0,len(a['Revenue'])),a['Revenue'],label='Revenue',linewidth=3,color='green')

# plt.bar(range(0,len(a['Marketing Spend'])),a['Marketing Spend'],label='Marketing Spend',linewidth=3,color='red')

# plt.plot(a['Visitors'],label='Visitors',linestyle='--',linewidth=3)

# plt.title('RAW DATA TO THIS',fontweight='bold')

# plt.xticks(range(0,len(a['Visitors'])),a['Date'],rotation=90)
# plt.legend(loc='upper right',fontsize=8)
# plt.xlabel('Date')
# plt.show()

# weight = .30
# a[['Revenue','Marketing Spend']].plot(kind='bar',color=['red','green'],figsize=(15,10),label=['Revenue','Marketing Spend'])
# a['Visitors'].plot(color='blue',secondary_y=True,linestyle='--',linewidth=2,label='Visitors')
# plt.title('RAW DATA TO THIS',fontweight='bold')
# plt.xticks(range(0,len(a['Visitors'])),a['Date'],rotation=90)
# plt.legend(loc='upper right',fontsize=8)
# plt.xlabel('Date')
# plt.xlim(-weight,len(a['Visitors'])-weight)
# plt.show()


# plt.figure(figsize=(10,6))
# plt.scatter(raw_data_read['Revenue'],raw_data_read['Marketing Spend'],color=['red'],marker='+')
# plt.show()

# groups = raw_data_read[['Promo','Revenue','Marketing Spend']].groupby('Promo')
# fig, ax = plt.subplots(figsize=(15,10))

# for promo,group in groups:
#     ax.scatter(group['Revenue'],group['Marketing Spend'],marker='x',label=promo)

# ax.legend()
# plt.show()

groups = raw_data_read[['Day_Name','Revenue','Marketing Spend']].groupby('Day_Name')
fig, ax = plt.subplots(figsize=(15,10))

for Day_name,group in groups:
    ax.scatter(group['Revenue'],group['Marketing Spend'],marker='x',label=Day_name)

ax.legend()
plt.show()