import csv, sqlite3
import pandas as pd
con = sqlite3.connect("RealWorldData.db")
cur = con.cursor()
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/FinalModule_Coursera_V5/data/ChicagoCensusData.csv')
%load_ext sql