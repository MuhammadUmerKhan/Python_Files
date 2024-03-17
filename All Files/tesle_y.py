
import requests
import yfinance as yf


url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/revenue.htm'

tesla_extract = requests.get(url)
tesla = yf.Ticker('TSLA')

tesla_data = tesla.history(period='max')

tesla_data.reset_index(inplace=True)
print(tesla_data.head())