# def dviding(a,b):
#     c=a/b
#     return c
# print(dviding(2,2))


def divisionError(Numerator,Dinimerator):
    try: 
        c = int(Numerator) /int(Dinimerator)
        return f'The Result is:{c} '
    except ZeroDivisionError:
       return "ZERO DIVISION ERROR"
    
Numerator=(input("Input Numerator: "))
Dinimerator=(input("Input Dinimerator: "))
result=divisionError(Numerator,Dinimerator)
print(result)