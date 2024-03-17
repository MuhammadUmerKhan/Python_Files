import pandas as pd
import requests
from bs4 import BeautifulSoup

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/amazon_data_webpage.html'
html_data = requests.get(url).text
# print(html_data)
soup = BeautifulSoup(html_data,'html.parser')
# print(soup.prettify())

# print(soup.find('title'))

amazon_data = pd.DataFrame(columns=['Date','Open','High','Low','Close','Volume'])

for row in soup.find('tbody').find_all('tr'):
    col = row.find_all('td')
    date = col[0]
    open = col[1]
    high = col[2]
    low = col[3]
    close = col[4]
    volume = col[5]

amazon_data = amazon_data.append({"Date":date, "Open":open, "High":high, "Low":low, "Close":close, "Volume":volume}, ignore_index=True)
print(amazon_data.head())
