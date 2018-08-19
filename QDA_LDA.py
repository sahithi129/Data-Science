import numpy as np
import matplotlib.pyplot as plt
import math
x = np.linspace(0,1,200)
y = np.zeros_like(x,dtype = np.int32)

x[0:100] = np.sin(4*np.pi*x)[0:100]
x[100:200] = np.cos(4*np.pi*x)[100:200]
y = 4*np.linspace(0,1,200)+0.3*np.random.randn(200)

m1 = [np.mean(x[0:100]), np.mean(y[0:100])]
m1 = np.array(m1)
   
m2 = [np.mean(x[100:200]), np.mean(y[100:200])]
m2 = np.array(m2)
var1 = np.linalg.inv(np.cov(x[0:100],y[0:100]))
var2 = np.linalg.inv(np.cov(x[100:200],y[100:200]))
var = np.linalg.inv(np.cov(x,y))

b_0 = -0.5*(np.dot(np.dot((m1+m2),var),(m1-m2).T))
b_1 = np.dot(var,(m1-m2).T)

y1 = []
x1 = np.random.uniform(-1,1,300)

y1 = -(b_0/b_1[1])-(x1)*(b_1[0]/b_1[1])

a = 0.5*(var1-var2)
print(a)
b = (np.dot(m1,var1)-np.dot(m2,var2))
#print(c)
c = (np.dot(np.dot(m1,var1),m1.T)-(np.dot(np.dot(m2,var2),m2.T)))
#print(b)
v1_det = np.linalg.det(np.cov(x[0:100],y[0:100]))
v2_det = np.linalg.det(np.cov(x[100:200],y[100:200]))
b0 = 0.5*((math.log(v1_det/v2_det))+c)
b11 = b[0]
b12 = b[1]
b33 = (a[0,1]+a[1,0])
print(b33)
b21 = a[0,0]
b22 = a[1,1]
y11 = []
y12 = []
for i in range(len(x1)):
    y11.append((-b12-b33*x1[i]-math.sqrt((b12+b33*x1[i])**2-4*b22*(b0+b11*x1[i]+b21*(x1[i]**2))))/2*b22)
    y12.append((-b12-b33*x1[i]+math.sqrt((b12+b33*x1[i])**2-4*b22*(b0+b11*x1[i]+b21*(x1[i]**2))))/2*b22)

label= np.ones_like(x)
label[0:100]=0
plt.plot(x1,y1)
plt.plot(x1,y12)
plt.plot(x1,y11)
#plt.ylim(-3,5)
plt.scatter(x,y,c=label)
plt.show()