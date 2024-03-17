import requests
# r=requests.get('https://xkcd.com/#')
# print(dir(r)) #200 means website response successfully
# print(help(r))
# print(r.text)
# print(r.ok)
# print(r.headers)
# downloading images from websites
img=requests.get(' https://imgs.xkcd.com/comics/gold.png')
# print(img.content)
with open('download_image.png','wb') as f:
    f.write(img.content)
# payload={'page':2,'count':25}
# r=requests.get('https://httpbin.org/get',params=payload )
# # print(r.text)
# print(r.url)


# posting data

# payload={'username':'umer','password':'trying'}
# r=requests.post('https://httpbin.org/post',data=payload )
# # print(r.text)
# print(r.text)
# print(r.json())
# r_dict=r.json()
# print(r_dict['form'])
# r=requests.get('https://httpbin.org/basic-auth/corey/testing',auth=('umer','trying') )
# print(r)

downloadind_img=requests.get('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0101EN-SkillsNetwork/labs/Module%205/data/Example1.txt')
print(downloadind_img.content)
with open('download_image1.png','wb') as downloading:
    downloading.write(downloadind_img.content)
