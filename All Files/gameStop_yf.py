import yfinance as yf
import pandas as pd
import requests
url ='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/revenue.htm'
gameStop_data = requests.get(url)
# print(gameStop_data)

gameStop=yf.Ticker('GME')

gme_data = gameStop.history(period='max')
gme_data.reset_index(inplace=True)
print(gme_data.head())
