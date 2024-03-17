# from randomuser import RandomUser
# import pandas as pd 
# r=RandomUser()
# some_list = r.generate_users(10)
# print(some_list)
from bs4 import BeautifulSoup
import requests 
html="<!DOCTYPE html><html><head><title>Page Title</title></head><body><h3><b id='boldest'>Lebron James</b></h3><p> Salary: $ 92,000,000 </p><h3> Stephen Curry</h3><p> Salary: $85,000, 000 </p><h3> Kevin Durant </h3><p> Salary: $73,200, 000</p></body></html>"
soup = BeautifulSoup(html, 'html5lib')
print(soup.prettify())
tag_object=soup.title
print("tag object:",tag_object)
print("tag object type:",type(tag_object))
tag_object=soup.h3
print(tag_object)