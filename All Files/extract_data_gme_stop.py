import yfinance as yf
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots

amd = yf.Ticker('AMD')

# def make_graph(stock_data, revenue_data, stock):
#     fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Historical Share Price", "Historical Revenue"), ertical_spacing = .3)
#     stock_data_specific = stock_data[stock_data.Date <= '2021--06-14']
#     revenue_data_specific = revenue_data[revenue_data.Date <= '2021-04-30']
#     fig.add_trace(go.Scatter(x=pd.to_datetime(stock_data_specific.Date, infer_datetime_format=True), y=stock_data_specific.Close.astype("float"), name="Share Price"), row=1, col=1)
#     fig.add_trace(go.Scatter(x=pd.to_datetime(revenue_data_specific.Date, infer_datetime_format=True), y=revenue_data_specific.Revenue.astype("float"), name="Revenue"), row=2, col=1)
#     fig.update_xaxes(title_text="Date", row=1, col=1)
#     fig.update_xaxes(title_text="Date", row=2, col=1)
#     fig.update_yaxes(title_text="Price ($US)", row=1, col=1)
#     fig.update_yaxes(title_text="Revenue ($US Millions)", row=2, col=1)
#     fig.update_layout(showlegend=False,
#     height=900,
#     title=stock,
#     xaxis_rangeslider_visible=True)
#     fig.show()
    

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
# gme_revenue["Revenue"] = gme_revenue['Revenue'].str.replace(',|\$',"")
print(gme_revenue)