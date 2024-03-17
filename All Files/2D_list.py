import numpy as nm
a=[[123],[456],[789]]
A=nm.array(a)
# print(A)
# print(A.ndim)
# print(A.shape)
# print(A.size) #Dimension
X=nm.array([[1,3],[3,1]])
Y=nm.array([[3,1],[1,3]])
# Z = X + Y
Z = X * Y
# print(Z)
F = 2*X
# print(F )
G = nm.dot(X,Y)
# print(G)

X=nm.array([[1,0,1],[2,2,2]]) 
out=X[0:2,2]
print(out)
