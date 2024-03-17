import json
import requests
import yfinance as yf
amd = yf.Ticker('AMD')  
data_url='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/data/amd.json'
scrap = requests.get(data_url).text

# with open('amd.json','w') as amd_write:
#     amd_data=amd_write.write(scrap)

with open('amd.json','r') as amd_read:
    amd_data=json.load(amd_read)

# print(amd_data['country'])
# print(amd_data['sector'])
amd_stock_data_share = amd.history(period='max')
print(amd_stock_data_share)