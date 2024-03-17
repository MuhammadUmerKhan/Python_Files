import imp
import pandas as pd
from pyparsing import col 
import yfinance as yf
import requests 
from bs4 import BeautifulSoup
import html5lib
url = ' https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/amazon_data_webpage.html'
try: 
    data = requests.get(url).text
except requests.exceptions.RequestException as e:
    print(e)
    data =""

soup = BeautifulSoup(data,'html.parser')
# print(soup)
data_list = []

for row in soup.find("tbody").find_all('tr'):
    col = row.find_all('td')
    date = col[0].text
    open = col[1].text
    high = col[2].text
    low = col[3].text
    close = col[4].text
    adj_close = col[5].text
    volume = col[6].text

    data_list.append({"Date":date,"Open":open,"High":high,"Low":low,"Close":close,"adj_close":adj_close,"Volume":volume})


amazon_data = pd.DataFrame(data_list)
print(amazon_data.tail(3))