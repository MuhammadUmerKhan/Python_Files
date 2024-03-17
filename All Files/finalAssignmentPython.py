from matplotlib import table
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots
tesla = yf.Ticker('TSLA')
tesla_data = tesla.history(period='max')
tesla_data.reset_index(inplace=True)
tesla_data.head()
url = 'https://www.macrotrends.net/stocks/charts/TSLA/tesla/revenue'
html_data = requests.get(url).text
soup = BeautifulSoup(html_data, 'html5lib')

tesla_revenue= pd.read_html(url, match="Tesla Quarterly Revenue", flavor='bs4')[0]
tesla_revenue=tesla_revenue.rename(columns = {'Tesla Quarterly Revenue(Millions of US $)': 'Date', 'Tesla Quarterly Revenue(Millions of US $).1': 'Revenue'}, inplace = False)
tesla_revenue["Revenue"] = tesla_revenue["Revenue"].str.replace(",","").str.replace("$","")
tesla_revenue.head()


gameStop = yf.Ticker('GME')
gme_data = gameStop.history(period='max')
gme_data.reset_index(inplace=True)
gme_data.head()
url1 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/stock.html'
html_data1 = requests.get(url1).text
soup1 = BeautifulSoup(html_data1, 'html5lib')

data_list1 = []
for table in soup1.find('tbody').find_all('tr'):
    col = table.find_all('td')
    date = col[0].text
    revenue = col[1].text
    
    data_list1.append({'Date':date,"Revenue":revenue})
    
gme_revenue = pd.DataFrame(data_list1)
gme_revenue.tail()

