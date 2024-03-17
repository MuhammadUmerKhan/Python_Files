ratings=[12,3,4,6,8,4,2,1306,4,0]
ratings.sort()
# print(ratings)
ratings.reverse()
# print(ratings)


class newClass(object):
    def __init__(self,text,type):
        self.text=text
        self.type=type
    def info(self,text,type):
        print(f'The Text is: {text}')
        print(f'The Type is: {type}')
result=newClass('Bold','Roman')
result.info('Bold','Roman')
# print(result)