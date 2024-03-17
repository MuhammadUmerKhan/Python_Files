import pandas as pd
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.xlsx'
df = pd.read_excel(url, sheet_name='Canada by Citizenship',
                       skiprows=range(20),
                        skipfooter=2)

df.head()
df.tail()
df.info(verbose=False)
df.columns
df.index
type(df.columns)
type(df.index)
df.columns.tolist()
df.index.tolist()
type(df.columns.tolist())
type(df.index.tolist())
df.shape
df[['AREA', 'REG', 'DEV', 'Type', 'Coverage']]
df.drop(['AREA', 'REG', 'DEV', 'Type', 'Coverage'], axis=1, inplace=True)
df.head(2)
df.rename(columns={'OdName':'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True)
df.columns
df.dtypes
df['Total'] = df.sum(axis=1)
df.isnull().sum()
df.describe()
df.set_index("Country", inplace=True)
df.head(3)
df.index.name = None
df.loc['Japan']
df.iloc[87]
df[df.index == 'Japan']
df.loc['Japan', 2013]
df.iloc[87, 36]
df.iloc[87, [3, 4, 5, 6, 7, 8]]
df.loc['Haiti']
df.loc['Haiti', 2000]
df.loc['Haiti', [1990, 1991, 1992, 1993, 1994, 1995]]
df.columns = list(map(str, df.columns))
years = list(map(str, range(1980, 2014)))
haiti = df.loc['Haiti', years]
condition = df['Continent'] == 'Asia'
df['Continent']
df[(df['Continent']=='Asia') & (df['Region']=='Southern Asia')]
df[(df['Continent']=='Africa') & (df['Region']=='Southern Africa')]
df.sort_values(by='Total', ascending=False, axis=0, inplace=True)