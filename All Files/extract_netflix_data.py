import html5lib
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
import requests

# Replace 'STOCK' with the actual stock symbol you want to retrieve data for
stock_symbol = 'NFLX'
stock = yf.Ticker(stock_symbol)

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/netflix_data_webpage.html'

try:
    data = requests.get(url).text
except requests.exceptions.RequestException as e:
    print("Error fetching data:", e)
    data = ""

soup = BeautifulSoup(data, 'html5lib')
# print(soup)
# Initialize an empty list to store data
data_list = []

for row in soup.find("tbody").find_all('tr'):
    col = row.find_all("td")
    date = col[0].text
    Open = col[1].text
    high = col[2].text
    low = col[3].text
    close = col[4].text
    adj_close = col[5].text
    volume = col[6].text

    # Append a dictionary to the list
    data_list.append({"Date": date, "Open": Open, "High": high, "Low": low, "Close": close, "Adj Close": adj_close, "Volume": volume})

# Create a DataFrame from the list of dictionaries
netflix_data = pd.DataFrame(data_list)

print(netflix_data)
