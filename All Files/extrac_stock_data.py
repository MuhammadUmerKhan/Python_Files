import yfinance as yf
import requests
import json
apple = yf.Ticker('AAPL')
url ='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/data/apple.json'
stock = requests.get(url).text
# print(stock )
# wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/data/apple.json
# with open('apple.json','w') as jsonFile:
#     apple_info = jsonFile.write(stock)
#     print(type(apple_info))

with open('apple.json') as readJson:
    apple_info = json.load(readJson)
    # print(apple_info)
    # print(apple_info['country'])

apple_share_price_data = apple.history(period="max")
# print(apple_share_price_data)
# print(apple_share_price_data.head())
# print(apple_share_price_data.reset_index(inplace=True))
# print(apple_share_price_data.plot(x="Date", y="Open"))

print(apple.dividends)
