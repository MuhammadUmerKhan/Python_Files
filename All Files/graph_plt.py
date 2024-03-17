import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


# increase size
plt.figure(figsize=[12,12])

# # data 
# a = np.array([1,2,10,3,0])
# b = np.array([2,2,2,2,2])
# c=np.array([2,4,3,1,2])

# a=np.arange(12) #arragne in 0 to 30
# b=np.random.randint(0,12,12) #randon arrange from 0 to 30


# plt.title('My First Excercise',fontsize='20',fontweight='bold')

# plt.plot(a,color='black',linestyle='-',label='line_1')
# plt.plot(b,linewidth=3,c='red',label='line_2')
# plt.plot(c,c='blue',linestyle='--',label='line_2')
# # plt.legend(loc='upper left') # loc is used to change the location of labels of line
# plt.xticks(range(0,len(a)),{'Jan','Feb','Mar','Apr','May','June','July','August','September','Octuber','November','December'})
# plt.xlabel('Months')
# plt.ylabel('Numbers')
# plt.show()


# Plots type
# plt.scatter(a,b)
# plt.show()

# bar plot
# category = ['Account A','Account B','Account C']
# revenue = np.array([300,500,800])

# plt.bar(range(0,3),revenue)
# plt.xticks(range(0,3),category)
# plt.show()

# subplots
# category = ['Account A','Account B','Account C']
# revenue = np.array([300,500,800])


# # bar
# plt.subplot(132)
# plt.bar(range(0,3),revenue)
# plt.xticks(range(0,3),category)
# # plt.show()


# # line
# plt.subplot(131)
# plt.plot(range(0,3),revenue)
# plt.xticks(range(0,3),category)


# # scatter
# plt.subplot(133)
# plt.scatter(range(0,3),revenue)
# plt.xticks(range(0,3),category)
# plt.show()

# Histogram
a=np.random.randn(10000)*100*100
print(a)
plt.hist(a,7)
plt.show()