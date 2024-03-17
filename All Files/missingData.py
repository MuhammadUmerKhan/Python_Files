import pandas as pd
import numpy as np
# import matplotlib.pylab as plt
import matplotlib as plt
from matplotlib import pyplot 


async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())
filename = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"

df = pd.read_csv(filename,header=None)

df.head(5)

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv(filename,names=headers)
df.head(5)

# replace "?" to NaN
df.replace("?",np.nan,inplace=True)
df.head()

missing_data = df.isnull()
missing_data.head(5)


for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")
    
# Calculate the mean value for the "normalized-losses
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses	 is: ",avg_norm_loss)

# Replace "NaN" with mean value in "normalized-losses" column
df["normalized-losses"].replace(np.nan,avg_norm_loss,inplace=True)

# Calculate the mean value for the "bore" column
avg_bore = df["bore"].astype("float").mean(axis=0)
print("Average of Bore is: ",avg_bore)

# Replace "NaN" with the mean value in the "bore" column
df["bore"].replace(np.nan,avg_bore,inplace=True)
df["bore"].head()

# df["stroke"]
# Q1 Based on the example above, replace NaN in "stroke" column with the mean value.
avg_stroke = df["stroke"].astype("float").mean(axis=0)
print("The Average of Stroke is: ",avg_stroke)

df["stroke"].replace(np.nan,avg_stroke,inplace=True)

# Calculate the mean value for the "horsepower" column
avg_horsepower = df["horsepower"].astype('float').mean(axis=0)
df["horsepower"].replace(np.nan,avg_horsepower,inplace=True)

# Calculate the mean value for "peak-rpm" column
avg_peak_rpm = df["peak-rpm"].astype('float').mean(axis=0)
df["peak-rpm"].replace(np.nan,avg_peak_rpm,inplace=True)


df['num-of-doors'].value_counts()
# We can see that four doors are the most common type. We can also use the ".idxmax()" method to calculate the most common type automatically:
df['num-of-doors'].value_counts().idxmax()
# The replacement procedure is very similar to what we have seen previously:
df["num-of-doors"].replace(np.nan,"four",inplace=True)
                           
#    Finally, let's drop all rows that do not have price data:
df.dropna(subset=["price"],axis=0,inplace=True)
# reset index, because we droped two rows
df.reset_index(drop=True,inplace=True)

df.head(15)



# Correct data format
df.dtypes
df[['bore','stroke']] = df[['bore','stroke']].astype('float')
df[['normalized-losses']] = df[['normalized-losses']].astype('int')
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
df.dtypes

df['city-L/100km'] = 235/df["city-mpg"]

# According to the example above, transform mpg to L/100km in the column of "highway-mpg" and change the name of column to "highway-L/100km".
df['highway-mpg'] = 235/df['highway-mpg']
df.rename(columns={'highway-mpg':'highway-L/100km'},inplace=True)

df.head()

# Data Normalization¶
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()

# According to the example above, normalize the column "height".
df['height'] = df['height']/df['height'].max()
df["horsepower"]=df["horsepower"].astype(int, copy=True)

# Binning¶
plt.pyplot.hist(df["horsepower"])
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")

bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
# bins
group_names = ['Low', 'Medium', 'High']
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
df[['horsepower','horsepower-binned']].head(20)

df["horsepower-binned"].value_counts()

pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")

pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")

df.columns

dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.head()

dummy_variable_1.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)
dummy_variable_1.head()

# merge data frame "df" and "dummy_variable_1" 
df = pd.concat([df, dummy_variable_1], axis=1)
# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)
df.head()

# Similar to before, create an indicator variable for the column "aspiration"
df['aspiration']
dummy_variable_2 = pd.get_dummies(df["aspiration"])
dummy_variable_2.head()

dummy_variable_2.rename(columns={'std':'aspiration-std','turbo':'aspiration-turbo'},inplace=True)
df=pd.concat([df,dummy_variable_2], axis = 1)
df.drop("aspiration",axis = 1, inplace = True)
df.head()

# Save the new csv:
df.to_csv('clean_df.csv')