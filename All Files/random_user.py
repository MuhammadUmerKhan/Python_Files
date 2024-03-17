# from randomuser import RandomUser
# import pandas as pd 
# r=RandomUser()
# some_list = r.generate_users(10)
# print(some_list)
# for user in some_list:
#     print(user.get_picture())
import pandas as pd
import requests
import json
data = requests.get("https://fruityvice.com/api/fruit/all")

results = json.loads(data.text)
# print(results[0]['id'])
df=pd.DataFrame(results)
print(df)
df2 = pd.json_normalize(results)
# print(df2['nutritions.fat'])