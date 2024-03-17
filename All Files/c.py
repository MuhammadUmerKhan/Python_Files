import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
arr = np.array([5,6,2,3])
arr1 = np.array([4,3,6,1])
arr2 = np.array([2,6,1,3])
arr3 = np.array([3,5,7,3])
df = pd.DataFrame({'Q1': arr, 'Q2': arr1, 'Q3': arr2, 'Q4': arr3})
df['Total'] = df.sum(axis=1)
print(df)
xlabels = df.columns[:-1]
ylabels = df.index
plt.plot(df.T)  # Transpose the DataFrame for correct plotting
plt.title('Student Graph')
plt.xticks(range(len(xlabels)), xlabels)
plt.yticks(range(len(ylabels)), ylabels)
plt.xlabel('Quarters')
plt.ylabel('Students')

plt.show()
# plt.ylim(0,)
