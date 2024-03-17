import pandas as pd
import numpy as np

df = pd.read_csv("dataset_part_3.csv")
df.head(3)
df['BoosterVersion'].unique()

df.isnull().sum()
# Identify and calculate the percentage of the missing values in each attribute
df.isnull().sum()/len(df)*100


df.dtypes

# TASK 1: Calculate the number of launches on each site
df.columns
df['LaunchSite'].value_counts()

# TASK 2: Calculate the number and occurrence of each orbit
df['Orbit'].value_counts()

# TASK 3: Calculate the number and occurence of mission outcome of the orbits
landing_outcomes = df['Outcome'].value_counts()
landing_outcomes
df['Outcome']

# ----True Ocean means the mission outcome was successfully landed to a specific 
#       region of the ocean while 
# ----False Ocean means the mission outcome was unsuccessfully landed to a 
#       specific region of the ocean. 
# ----True RTLS means the mission outcome was successfully landed to a ground pad
# ----False RTLS means the mission outcome was unsuccessfully landed to a ground 
#       pad.True ASDS means the mission outcome was successfully landed to a drone ship 
# ----False ASDS means the mission outcome was unsuccessfully landed to a drone ship. 
# ----None ASDS and None None these represent a failure to land.

landing_outcomes.keys()

for i, outcome in enumerate(landing_outcomes.keys()):
    print(i, outcome)
    
# We create a set of outcomes where the second stage did not land successfully:
bad_outcomes = set(landing_outcomes.keys()[[1, 3, 5, 6, 7]])
bad_outcomes

# TASK 4: Create a landing outcome label from Outcome column
landing_class = []
for outcome in df['Outcome']:
    if outcome in bad_outcomes:
        landing_class.append(0)
    else:
        landing_class.append(1)

df['Class'] = landing_class
df['Class'].value_counts()
df[['Class']].head(8)

df['Class'].mean()

df.head()

df.to_csv("dataset_part_2.csv", index=False)