i=[]
for bscs in range(0,100):
    i.extend('BSCS\n')

 
 
with open('bscs.txt','w') as bscs_write:
    bscs_write.write(''.join(i))