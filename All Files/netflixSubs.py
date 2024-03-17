import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

raw_data_read = pd.read_csv('D:\Marketing Raw Data.csv')


first_10_data = raw_data_read.tail(10).reset_index()
print(first_10_data)
# plt.figure(figsize=[10,10])
# plt.bar(range(0,len(first_10_data['Marketing Spend'])),first_10_data['Marketing Spend'],linestyle='--',linewidth=3,color='red',label='Marketing Spend')
# plt.xticks(range(0,len(first_10_data['Marketing Spend'])),first_10_data['Date'],rotation = 90)
# plt.legend(loc='upper right',fontsize=8)
# plt.xlabel('Date')
# plt.title('Graph Sheet')
# plt.show() 

# weight = .30
# first_10_data[['Revenue','Visitors']].plot(kind='bar',color=['green','purple'],figsize=(15,10),label=['Revenue','Visitors',])
# first_10_data['Marketing Spend'].plot(color='blue',label='Marketing Spend',linestyle='--')
# plt.xticks(range(0,len(first_10_data['Visitors'])),first_10_data['Date'],rotation=90)
# plt.title('Revenue_Visitors_Marketing Spend')
# plt.xlabel('DATE')
# plt.xlim(-weight,len(first_10_data['Visitors']-weight))
# plt.legend()
# plt.show()

# groups = raw_data_read[['Promo','Revenue','Marketing Spend']].groupby('Promo')
# fig, ax = plt.subplots(figsize=(15,10))

# for promo,group in groups:
#     ax.scatter(group['Revenue'],group['Marketing Spend'],marker='x',label=promo)

# ax.legend()
# plt.show()


