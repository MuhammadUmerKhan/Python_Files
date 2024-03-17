from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


df = pd.read_csv("C:\DATA SCIENCE\Python-git-files\All Files\ML\predictive_maintenance.csv")
df.head()
df.shape
df.describe()
df.columns
df['Failure Type'].unique()
df.isnull().sum()

df.drop(columns=['UDI', 'Product ID'], inplace=True)
df.shape

product_type = df['Type'].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(12, 7))

# First subplot: Bar plot
sns.barplot(x=product_type.index, y=product_type.values, ax=axes[0], palette="coolwarm")
axes[0].set_title("Product Type", fontsize=20, color='Red', fontname='Times New Roman')
axes[0].grid(True)

# Second subplot: Pie chart
axes[1].pie(product_type, autopct='%1.2f%%')
axes[1].set_title("Product Type", fontsize=20, color='Red', fontname='Times New Roman')
axes[1].legend(product_type.index, loc="best")  # Add legend
axes[1].grid(True)

# Show the plots
plt.show()

# Plot for failure type
failure_type = df['Failure Type'].value_counts()
sns.barplot(x=failure_type.index, y=failure_type.values)
plt.title("Failure Type", fontsize=20, color='Red', fontname='Times New Roman')
plt.grid(True)
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(20, 10))
sns.barplot(data=df, x=df['Failure Type'], y=df['Air temperature [K]'], palette='coolwarm', ax=axes[0, 0])
axes[0, 0].legend(loc='upper right', bbox_to_anchor=(1, 1))
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45, ha='right')

sns.barplot(data=df, x=df['Failure Type'], y=df['Process temperature [K]'], palette='coolwarm', ax=axes[0, 1])
axes[0, 1].legend(loc='upper right', bbox_to_anchor=(1, 1))
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha='right')

sns.barplot(data=df, x=df['Failure Type'], y=df['Rotational speed [rpm]'], palette='coolwarm', ax=axes[1, 0])
axes[1, 0].legend(loc='upper right', bbox_to_anchor=(1, 1))
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right')

sns.barplot(data=df, x=df['Failure Type'], y=df['Torque [Nm]'], palette='coolwarm', ax=axes[1, 1])
axes[1, 1].legend(loc='upper right', bbox_to_anchor=(1, 1))
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()

df.columns
# failure_type = df['Failure Type'].value_counts()
plt.figure(figsize=(12, 7))
sns.barplot(data=df, x=df['Tool wear [min]'], y=df['Failure Type'], errorbar=None, palette="coolwarm")
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

dfcat = df.groupby('Type')['Failure Type'].value_counts().unstack().fillna(0)
plt.figure(figsize=(12, 2))
sns.heatmap(data=dfcat, annot=True)
plt.show()

df.info()
# corr = df.select_dtypes(include=['int', 'float'])
# corr.corr()
# plt.figure(figsize=(12, 7))
# sns.heatmap(data=corr, annot=True)
# plt.show()

plt.figure(figsize=(12, 3))
plt.subplot(1, 5, 1)
sns.histplot(df['Air temperature [K]'], color='blue', bins=20, edgecolor='black', alpha=0.7)
plt.title('Air Temperature')
plt.grid()

plt.subplot(1, 5, 2)
sns.histplot(df['Process temperature [K]'], color='green', bins=20, edgecolor='black', alpha=0.7)
plt.title('Process Temperature')
plt.grid()

plt.subplot(1, 5, 3)
sns.histplot(df['Rotational speed [rpm]'], color='olive', bins=20, edgecolor='black', alpha=0.7)
plt.title('Rotational Temperature')
plt.grid()

plt.subplot(1, 5, 4)
sns.histplot(df['Torque [Nm]'], color='yellow', bins=20, edgecolor='black', alpha=0.7)
plt.title('Toque')
plt.grid()

plt.subplot(1, 5, 5)
sns.histplot(df['Tool wear [min]'], color='orange', bins=20, edgecolor='black', alpha=0.7)
plt.title('Tool Wear')
plt.grid()

plt.tight_layout()
plt.show()

# for i in df.columns:
#     if df[i].dtype != 'object':
#         z_score = (df[i] - df[i].mean()) / df[i].std()
#         df = df[(z_score > -3) & (z_score < 3)]

df.shape # (9529, 11)
df.head()
# df.drop(columns='Z_score', inplace=True)

le = LabelEncoder()
# df['Type'] = le.fit_transform(df['Type']) # M=2, L=1, H=0   
# df.head()
# df['Type'].unique()

df['Failure Type_encoded'] = le.fit_transform(df['Failure Type'])
df['Failure Type'].value_counts()
df['Failure Type_encoded'].value_counts()
df.to_csv("model_data.csv", index=False)
df.columns

df_encoded = pd.get_dummies(df, columns=['Type'], prefix='Type', drop_first=True)
# df_encoded = pd.get_dummies(df, columns=['Type'])
df_encoded.head()

df_encoded[['Type_L', 'Type_M']] = df_encoded[['Type_L', 'Type_M']].astype('int')

df_encoded.info()
df_encoded['Failure Type_encoded'].value_counts()

numeric_data = df.select_dtypes(include=['int64', 'int32', 'float64'])
numeric_data.head()
corr = numeric_data.corr()

sns.heatmap(corr, annot=True)

X = df_encoded.drop(columns=['Target', 'Failure Type', 'Failure Type_encoded'])
Y = df_encoded['Failure Type_encoded']
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(x_train_scaled, Y, test_size=0.2, random_state=42)
x_train.shape
x_test.shape

model_random = RandomForestClassifier(random_state=42)
model_random.fit(x_train, y_train)
model_random.score(x_test, y_test)*100 # 98.25
prediction = model_random.predict(x_test)
score = cross_val_score(RandomForestClassifier(random_state=42), x_train_scaled, Y, cv=5)
np.mean(score) # 0.8937000000000002

cm = confusion_matrix(prediction, y_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

xgb_model = XGBClassifier()
xgb_model.fit(x_train, y_train)
xgb_model.score(x_test, y_test)*100 # 98.25
prediction_xgb = xgb_model.predict(x_test)
score = cross_val_score(XGBClassifier(), x_train_scaled, Y, cv=5)
np.mean(score)

cm = confusion_matrix(prediction_xgb, y_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()