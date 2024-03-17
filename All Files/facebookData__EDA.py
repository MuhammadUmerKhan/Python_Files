from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import math

df = pd.read_csv("C:\DATA SCIENCE\Python-git-files\dataset\pseudo_facebook.csv\pseudo_facebook.csv")
df.head()
df.shape
df.isnull().sum()
df.info()

df['tenure'].median()

df.dropna(inplace=True)

df.duplicated().sum()

df.head()
df['age'].sort_values(ascending=False)
df['age'].sort_values(ascending=True)

age_grp = ['10-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100', '101-110', '111-120']
df['age_grp'] = pd.cut(df['age'], bins=np.arange(10, 121, 10), labels=age_grp, right=True)
df.head()
df['age_grp']

age_grp = df['age_grp'].value_counts().reset_index()
age_grp

# visualize age_grp 
plt.figure(figsize=(20, 10))
sns.pointplot(x='age_grp', y='count', data=age_grp)

df.head()

age_20s = df[(df['age']>20) & (df['age']<30)] 
df_20s = age_20s['age'].sort_values(ascending=True).value_counts().reset_index()

fig = px.bar(df_20s,
    x='age', y='count',
    color='age'
)
fig.update_layout(
    xaxis_title = 'Age', yaxis_title = 'Count',
    title = 'Ages', height=400, width=800
)
fig.show()

df.columns
df_gender = df['gender'].value_counts().reset_index()
df_gender

fig = px.pie(df_gender, names=df_gender['gender'], values=df_gender['count'], title='Gender Percentage')
fig.update_layout(
    title=dict(text='Gender Percentage', x=0.5, y=0.95, font=dict(size=20)),
    legend=dict(title='', x=0.8, y=0.5),
    margin=dict(t=50, b=50, r=50, l=50),
    paper_bgcolor='white',
    showlegend=True,
)
fig.show()

fig = px.bar(df_gender,
    x='gender', y='count',
    color='gender'
)
fig.update_layout(
    xaxis_title = 'Gender', yaxis_title = 'Count',
    title = 'Gender Count', height=400, width=800
)
fig.show()

age_gender_group = df[['age_grp', 'gender']].value_counts().reset_index()
age_gender_group

fig = px.bar(age_gender_group,
             x='age_grp', y='count', color='gender',
             barmode='group',
             title='Age Group Distribution by Gender', 
             labels={'count': 'Count', 'age_group': 'Age Group'})
fig.update_layout(
    xaxis=dict(title='Age Group'),
    yaxis=dict(title='Count'),
    legend=dict(title='Gender'),
    margin=dict(t=50, b=50, r=50, l=50),
    paper_bgcolor='white',
)
fig.show()

age_gender_group_=  age_20s[['age', 'gender']].value_counts().sort_values(ascending=False).reset_index()
age_gender_group_

fig = px.bar(age_gender_group_,
             x='age', y='count', color='gender',
             barmode='group',
             title='Age Group Distribution by Gender Age(20-30)', 
             labels={'count': 'Count', 'age_group': 'Age Group'})
fig.update_layout(
    xaxis=dict(title='Age Group'),
    yaxis=dict(title='Count'),
    legend=dict(title='Gender'),
    margin=dict(t=50, b=50, r=50, l=50),
    paper_bgcolor='white',
)
fig.show()

age_10s = df[(df['age']>10) & (df['age']<20)] 
age_10s
df_10s = age_10s['age'].sort_values(ascending=True).value_counts().reset_index()
plt.figure(figsize=(20, 10))
sns.countplot(y='age', data=df_10s)

fig = px.bar(df_10s,
    x='count', y='age',
    color='age', orientation='h'
)
fig.update_layout(
    xaxis_title = 'Count', yaxis_title = 'age',
    title = 'Ages', height=400, width=800
)
fig.show()
# df
plt.figure(figsize=(20, 10))
sns.barplot(x='age_grp', y='likes_received', data=df, hue='gender')

plt.figure(figsize=(20, 10))
sns.barplot(x='age_grp', y='likes', data=df, hue='gender')

plt.figure(figsize=(12, 8))
sns.barplot(x='age', y='likes', data=age_10s, hue='gender')

gender_friend = df.groupby('gender')['friend_count'].sum().reset_index()
fig = px.bar(gender_friend,
    x='friend_count', y='gender',
    color='friend_count', orientation='h'
)
fig.update_layout(
    xaxis_title = 'Friend Count', yaxis_title = 'Gender',
    title = 'Female vs Male Friends', height=400, width=800
)
fig.show()

fig = px.pie(gender_friend, names=gender_friend['gender'], values=gender_friend['friend_count'], )
fig.update_layout(
    title=dict(text='Female vs Male Friends', x=0.5, y=0.95, font=dict(size=20)),
    legend=dict(title='', x=0.8, y=0.5),
    margin=dict(t=50, b=50, r=50, l=50),
    paper_bgcolor='white',
    showlegend=True,
)
fig.show()

