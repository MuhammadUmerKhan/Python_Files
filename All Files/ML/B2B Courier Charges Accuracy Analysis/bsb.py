import pandas as pd

order_report = pd.read_csv("./b2b/Order Report.csv")
sku_master = pd.read_csv('./b2b/SKU Master.csv')
pincode_mapping = pd.read_csv('./b2b/pincodes.csv')
courier_invoice = pd.read_csv('./b2b/Invoice.csv')
courier_company_rates = pd.read_csv('./b2b/Courier Company - Rates.csv')

order_report.head()
sku_master.head()
pincode_mapping.head()
courier_invoice.head()
courier_company_rates.head()

df = [order_report, sku_master, pincode_mapping, courier_invoice, courier_company_rates]
for table in df:
    # print(table)
    print(table.isnull().sum())
order_report.drop(columns=['Unnamed: 3', 'Unnamed: 4'], inplace=True)
sku_master.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
pincode_mapping.drop(columns=['Unnamed: 3', 'Unnamed: 4'], inplace=True)

order_report.head()
merged_data = pd.merge(order_report, sku_master, on='SKU')
merged_data.head()

merged_data = merged_data.rename(columns={'ExternOrderNo': 'Order ID'})
merged_data.head()
