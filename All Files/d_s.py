from bs4 import BeautifulSoup
import requests
# url = "http://www.ibm.com"
# data  = requests.get(url).text 
# soup = BeautifulSoup(data,"html.parser")  # create a soup object using the variable 'data'
# for link in soup.find_all('a',href=True):  # in html anchor/link is represented by the tag <a>
#     (link.get('href'))
    
    
# # Images links

# for link in soup.find_all('img'):
#     print(link)
#     print(link.get('src'))

#The below url contains an html table with data about colors and color codes.
# url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DA0321EN-SkillsNetwork/labs/datasets/HTMLColorCodes.html"
# data  = requests.get(url).text
# soup = BeautifulSoup(data,"html.parser")
# print(soup)
# table_finding=soup.find('table')
# print(table_finding)
# print(table_finding.find_all(id='boldest'))
# for