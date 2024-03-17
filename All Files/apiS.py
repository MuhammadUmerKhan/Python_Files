# import pandas as pd 
# list_={'a':[1,2,3],'b':[4,5,6]}
# data_frame=pd.DataFrame(list_)
# print(data_frame.head())
# print(data_frame.mean())

# def one_dict(list_dict):
#     keys=list_dict[0].keys()
#     out_dict={key:[] for key in keys}
#     for dict_ in list_dict:
#         for key, value in dict_.items():
#             out_dict[key].append(value)
#     return out_dict
# import pandas as pd
# import matplotlib.pyplot as plt
# df=pd.DataFrame(dict_)
# type(df)

import requests
import pandas as pd
filename = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/Chapter%205/Labs/Golden_State.pkl"

def download(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)

download(filename, "Golden_State.pkl")
file_name = "Golden_State.pkl"
games = pd.read_pickle(file_name)
print(games.head())
# print()