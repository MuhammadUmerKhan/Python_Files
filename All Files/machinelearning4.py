import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
df = pd.read_csv(path)
df.head()
df.shape

x = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
x[:,1] = le_sex.transform(x[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
x[:,2] = le_BP.transform(x[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
x[:,3] = le_Chol.transform(x[:,3])

y = df["Drug"]
# Setting up Decision Tree
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)
x_train.shape
y_train.shape

print("Shape of xTrain set is {}".format(x_train.shape), "and", "Size of xTrain set is {}".format(y_train.shape))

x_test.shape
y_test.shape

# Modeling
drugTree = DecisionTreeClassifier(criterion='entropy', max_depth=4)
drugTree.fit(x_train, y_train)

# Prediction 
predTree = drugTree.predict(x_test)
predTree[0:5]
y_test[0:5]

# Evaluation
print("DecisionTree's Accuracy: {}".format(metrics.accuracy_score(y_test, predTree)))

# Visualization
tree.plot_tree(drugTree)
plt.show()