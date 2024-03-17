# with open('Example.txt','w') as newFile:
#     content=['This is Line 1\nThis is Line 2\nThis is Line 3']
#     fileContent=newFile.writelines(content)

# with open('Example.txt','r') as new:
#     with open('Example1.txt','w') as reading:
#         readindContent=reading.writelines(new)

# with open('Progrress.txt','w') as progress:
#     contentWrite=['Abdur Rafay,80,90,80,98\nRayyan,89,89,88,99\nUmer,100,99,89,78']
#     writingContent=progress.writelines(contentWrite)
data1=['\n------------Result------------\n']
with open('Progrress.txt','r') as opening:

    while True:
        readingContent=opening.readline()
        if not readingContent:
            break
        marks0=readingContent.split(',')[0]
        marks1=readingContent.split(',')[1]
        marks2=readingContent.split(',')[2]
        marks3=readingContent.split(',')[3]
        marks4=readingContent.split(',')[4]
        marks=[marks0,marks1,marks2,marks3,marks4]
        def result(marks1,marks2,marks3,marks4):
            finalResult= ((int(marks1) + int(marks2) + int(marks3) + int(marks4))/400)*100
            # divide=add/400
            # finalResult=divide*100
            return f'{finalResult}'
        now=result(marks1,marks2,marks3,marks4)
        print1=f'Percentage of {marks[0]} is: {now}%'
        data1.extend([print1,'\n'])
        
# print(readingContent)


# with open('Result.txt','a') as writing:
#     writing.write(''.join(data1))
data=[]
with open('Progrress.txt','r') as reading:
        while True:
            nowReads=reading.readline()
            if not nowReads:
                break
            m1= (f'The Marks of {nowReads.split(",")[0]} in Math is: {nowReads.split(",")[1]}')
            m2=(f'The Marks of {nowReads.split(",")[0]} in English is: {nowReads.split(",")[2]}')
            m3=(f'The Marks of {nowReads.split(",")[0]} in Physics is: {nowReads.split(",")[3]}')
            m4=(f'The Marks of {nowReads.split(",")[0]} in Islamiat is: {nowReads.split(",")[4]}')
   
            data.extend([m1,'\n',m2,'\n',m3,'\n',m4,'\n'])
            # print(m1)
            # print(m2)
            # print(m3)
            # print(m4)
            # print(data)
with open('Result.txt','w') as resultStudents:
        resultStudents.write(''.join(data))
        resultStudents.write(''.join(data1))









# data=['\n------------Result------------\n']
# with open('Progrress.txt','r') as opening:

#     while True:
#         readingContent=opening.readline()
#         if not readingContent:
#             break
#         marks0=readingContent.split(',')[0]
#         marks1=readingContent.split(',')[1]
#         marks2=readingContent.split(',')[2]
#         marks3=readingContent.split(',')[3]
#         marks4=readingContent.split(',')[4]
#         marks=[marks0,marks1,marks2,marks3,marks4]
#         def result(marks1,marks2,marks3,mark4):
#             add= int(marks1) + int(marks2) + int(marks3) + int(marks4)
#             divide=add/400
#             finalResult=divide*100
#             return f'{finalResult}'
#         now=result(marks1,marks2,marks3,marks4)
#         print1=f'Percentage of {marks[0]} is: {now}%    '
#         data.extend([print1,'\n'])
        
# # print(readingContent)


# with open('Result.txt','a') as writing:
#     writing.write(''.join(data))
