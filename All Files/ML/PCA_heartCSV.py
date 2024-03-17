from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Load heart disease dataset in pandas dataframe
df = pd.read_csv("heart.csv")
df.head()

df.describe()
df.shape
df.isnull().sum()
df.info()
for i in df.columns:
    if df[i].dtype == 'object':
        print(i+':', df[i].unique())

for i in df.columns:
    if df[i].dtype != 'object':
        print(i)

df.head()
min(df.Age)
# Treat Outliers
# Cholesterol outlier
z_score = (df['Cholesterol'] - df['Cholesterol'].mean())/df['Cholesterol'].std()
df['Z_score'] = z_score
df1 = df[(df['Z_score']>-3) & (df['Z_score']<3)]
df1.shape
df1.head()
df1.drop(columns='Z_score', inplace=True)
df.drop(columns='Z_score', inplace=True)
# MaxHR
z_score = (df1['MaxHR'] - df1['MaxHR'].mean())/df1['MaxHR'].std()
df1['Z_score'] = z_score
df2 = df1[(df1['Z_score']>-3) & (df1['Z_score']<3)]
df2.shape
df2.drop(columns='Z_score', inplace=True)

# RestingBP
z_score = (df2['RestingBP'] - df2['RestingBP'].mean())/df2['RestingBP'].std()
df2['Z_score'] = z_score
df3 = df2[(df2['Z_score']>-3) & (df2['Z_score']<3)]
df3.shape
df3.drop(columns='Z_score', inplace=True)
df3.head()
# FastingBS
z_score = (df3['FastingBS'] - df3['FastingBS'].mean())/df3['FastingBS'].std()
df3['Z_score'] = z_score
df4 = df3[(df3['Z_score']>-3) & (df3['Z_score']<3)]
df4.shape
df4.drop(columns='Z_score', inplace=True)
df4.head()

# Oldpeak
z_score = (df4['Oldpeak'] - df4['Oldpeak'].mean())/df4['Oldpeak'].std()
df4['Z_score'] = z_score
df5 = df4[(df4['Z_score']>-3) & (df4['Z_score']<3)]
df5.shape
df5.drop(columns='Z_score', inplace=True)
df5.head()


# No outlier
df5.to_csv("heart_No_Outlier.csv", index=False)
df = pd.read_csv("heart_No_Outlier.csv")
df.head()
df.shape
df.describe()
df.info()
df.dtypes
df.columns
df.head()

le = LabelEncoder()
df.head()

df['RestingECG'] = le.fit_transform(df['RestingECG'])
df.head()

df['ST_Slope'] = le.fit_transform(df['ST_Slope'])
df.head()
df['ExerciseAngina'] = le.fit_transform(df['ExerciseAngina'])
df2 = pd.get_dummies(df, drop_first=True)
df2.head()
df2[['ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA', 'Sex_M']].dtypes
df2[['ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA', 'Sex_M']] = df2[['ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA', 'Sex_M']].astype('int')
df2.head()

X = df2.drop(columns='HeartDisease', axis='columns')
Y = df2.HeartDisease

X.head()

X_scaled = StandardScaler().fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=4)

# logistic regression
logistic = LogisticRegression(solver='liblinear')
logistic.fit(x_train, y_train)
logistic.score(x_test, y_test) # 0.88888888888888888

random = RandomForestClassifier(n_estimators=40)
random.fit(x_train, y_train)
random.score(x_test, y_test) # 0.894444444444

# SVM
svc = SVC()
svc.fit(x_train, y_train)
svc.score(x_test, y_test) # 0.894444444

pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, Y, test_size=0.2, random_state=4)

logistic = LogisticRegression(solver='liblinear')
logistic.fit(X_train_pca, y_train)
logistic.score(X_test_pca, y_test) # 0.9055555555

# Random Forest
random = RandomForestClassifier(n_estimators=40)
random.fit(X_train_pca, y_train)
random.score(X_test_pca, y_test) # 0.894444444444

# SVM
svc = SVC()
svc.fit(X_train_pca, y_train)
svc.score(X_test_pca, y_test) # 0.75

# cross val score
log = cross_val_score(LogisticRegression(solver='liblinear', multi_class='ovr'), X_pca, Y, cv=10)
np.average(log) # 0.829775
log1 = cross_val_score(LogisticRegression(solver='liblinear', multi_class='ovr'), X, Y, cv=10)
np.average(log1) # 0.8353183

random1 = cross_val_score(RandomForestClassifier(n_estimators=50), X, Y, cv=10)
np.average(random1) # 0.84196
random2 = cross_val_score(RandomForestClassifier(n_estimators=50), X_pca, Y, cv=10)
np.average(random2) # 0.82079

svc1 = cross_val_score(SVC(gamma='auto'), X, Y, cv=10)
np.average(svc1) # 0.565

svc2 = cross_val_score(SVC(gamma='auto'), X_pca, Y, cv=10)
np.average(svc2) # 0.5539