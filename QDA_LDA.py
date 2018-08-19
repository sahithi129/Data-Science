# Decision Boundary of LDA and QDA

import numpy as np
import matplotlib.pyplot as plt
import math

#input 
x = np.linspace(0,1,200)
y = np.zeros_like(x,dtype = np.int32)

x[0:100] = np.sin(4*np.pi*x)[0:100]
x[100:200] = np.cos(4*np.pi*x)[100:200]
y = 4*np.linspace(0,1,200)+0.3*np.random.randn(200)

# Mean of CLASS 0
m1 = np.array([np.mean(x[0:100]), np.mean(y[0:100])])
# Mean of CLASS 1   
m2 = np.array([np.mean(x[100:200]), np.mean(y[100:200])])

# Variance Inverse of CLASS 0
var1 = np.linalg.inv(np.cov(x[0:100],y[0:100]))

# Variance Inverse of CLASS 1
var2 = np.linalg.inv(np.cov(x[100:200],y[100:200]))

# Variance Inverse of both classes
var = np.linalg.inv(np.cov(x,y))

# Solving Linear equation of Linear Discriminant Analysis where var1=var2=var
b_0 = -0.5*(np.dot(np.dot((m1+m2),var),(m1-m2).T))
b_1 = np.dot(var,(m1-m2).T)


# X coordinates 
#x1= np.arange(-1.0, 1.0, 0.01)
x1 = np.linspace(-3,3,200)
# y coordinates obtained from linear equation 
y1 = []
y1 = -(b_0/b_1[1])-(x1)*(b_1[0]/b_1[1])

"""Quadratic Discriminant Analysis where var1, var2, var are not equal
    we will get quadratic equation b0+b11x+b12y+b33xy+b21x^2+b33y^2=0 
    we will solve for y """

a = -0.5*(var1-var2)

b = (np.dot(m1,var1)-np.dot(m2,var2))

c = (0.5*(np.dot(np.dot(m1,var1),m1.T)-(np.dot(np.dot(m2,var2),m2.T))))

v1_det = np.linalg.det(np.cov(x[0:100],y[0:100]))
v2_det = np.linalg.det(np.cov(x[100:200],y[100:200]))

b0 = -(0.5*(math.log(v1_det/v2_det))+c) # constant value in equation
b11 = b[0] # coefficient of x
b12 = b[1] # coefficient of y
b33 = (a[0,1]+a[1,0]) # coefficient of xy
b21 = a[0,0] # coefficient of x^2
b22 = a[1,1] # coefficient of y^2

sqrt= np.sqrt(((b12+b33*x1)**2)-4*b22*(b0+b11*x1+b21+x1**2))

y12= (-b12-b33*x1-sqrt)/(2*b22) # solution 1 for y of quadratic equation
y11= (-b12-b33*x1+sqrt)/(2*b22) # solution 2 for y of quadratic equation

# class labels
label= np.ones_like(x)
label[0:100]=0

# plotting decision boundaries
plt.xlim(-2, 2) 
plt.ylim(-1, 6)
print('For noise = 0.3')
plt.plot(x1,y1,label = 'LDA',color = 'blue')
plt.plot(x1,y12, label = 'QDA', color='red')
plt.plot(x1,y11)
plt.scatter(x,y,c=label)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Boundary of LDA and QDA')
plt.legend()
plt.show()





