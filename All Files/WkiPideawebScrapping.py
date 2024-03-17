from bs4 import BeautifulSoup
import requests
import pandas as pd
# url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_Indian_films"
# req = requests.get(url)
# soup = BeautifulSoup(req.text, 'html')

# table = soup.find_all('table')[1]
# title = table.find('tr')
# heads = [titles.text.strip() for titles in title]
# sorted_head = [sorted for sorted in heads if sorted != '']
# df =pd.DataFrame(columns = sorted_head)
# df = df.drop(columns='Title', axis=1)
# columns_data = table.find_all('tr')

# for row in columns_data[1:]:
#     row_data = row.find_all('td')
#     individual_data = [data.text.strip() for data in row_data]
#     length = len(df)
#     df.loc[length] = individual_data
    
# df = df.drop(columns="Reference(s)")
# df.head()

url = "https://en.wikipedia.org/wiki/World_population"
req = requests.get(url)
soup = BeautifulSoup(req.text, 'html')

table = soup.find_all('table')[4]
title = table.find('tr')
heads = [titles.text.strip() for titles in title]
sorted_head = [sorted for sorted in heads if sorted != '']
df =pd.DataFrame(columns = sorted_head)
df.columns
columns_data = table.find_all('tr')
for row in columns_data[2:]:
    row_data = row.find_all('td')
    individual_data = [data.text.strip() for data in row_data]
    # print(individual_data)
    length = len(df)
    df.loc[length] = individual_data

df.head()  