# with open('Result.txt','r') as reading:
#     while True:
#         readingContent=reading.readline()
#         if not readingContent:
#             break
#     print(readingContent)


import pandas as pd

prophetNames={'Name':['Prophet Noh','Prophet Moosa','Prophet Isa','Prophet Mohammad','Prophet Ibrahim'],
      'Age':[950,125,'Alive',63,195],
      'Gender':['Male','Male','Male','Male','Male']}
df=pd.DataFrame(prophetNames)
print(df)

studentsProgression={'Students':['Abdullah','Owais','AbdurRafay','Rayyan','Zaid'],
                     'Age':[15,14,14,13,14],
                     'Country':['Pakistan','Pakistan','Pakistan','Pakistan','Pakistan'],
                     'Percentage':[81,84,82,83,85]}
df1=pd.DataFrame(studentsProgression)
print(df1)


# print(df.iloc[0,0])



