import select
import pandas as pd
import numpy as np
import sqlite3
import sqlmagic


con  = sqlite3.connect("my_data1.db")
cur = con.cursor()

# %load_ext sql
# %sql sqlite:///my_data1.db

df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/data/Spacex.csv")
df.to_sql("SPACEXTBL", con, if_exists='replace', index=False,method="multi")

# %sql create table SPACEXTABLE as select * from SPACEXTBL where Date is not null

# %sql select * from SPACEXTABLE
df.head()
df.shape
# df.to_sql("SPACEXTBL.sql", con)