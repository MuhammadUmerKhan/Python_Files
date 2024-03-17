import pandas as pd 
readCSV = pd.read_csv('D:\owid-covid-data.csv')
read1oData = readCSV.head(5).reset_index()
print(readCSV)
