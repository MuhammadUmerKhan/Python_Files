import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf 

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/stock.html'
gme_data = requests.get(url).text

gme_data_parser = BeautifulSoup(gme_data,'html.parser')


data_list = []
for row in gme_data_parser.find('tbody').find_all('tr'):
    col = row.find_all('td')
    date = col[0]
    revenue = col[1]

    data_list.append({'Date':date,'Revenue':revenue})


gme_revenue = pd.DataFrame(data_list)
gme_revenue["Revenue"] = gme_revenue['Revenue'].str.replace(',|\$',"")
print(gme_revenue)