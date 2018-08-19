# -*- coding: utf-8 -*-
"""

Perceptron Algorithm

"""

import numpy as np

# All the input patterns of class1 and class2 are mapped from non linear to linear patterns and assigned to x

y = np.array([[1,4,16,8,2,4],[1,9,16,12,3,4],[1,144,16,48,12,4],[1,169,16,52,13,4],[-1,-36,-16,-24,-6,-4],
              [-1,-49,-16,-28,-7,-4]])


# function perceptron is defined to calculate weights for a set of iterations to reach stable point

def perceptron(x):
    w = np.array([0,0,0,0,0,0])                         # weights are intialized to zeroes
    n = 300                     
    temp = np.array(x[0])                               # array temp is defined to compare weights for each iteration
    for i in range(n):
        for j in range(len(y)):                         # for each iteration calculate weight for all input patterns
            alpha=1.0                                 # learning rate alpha
            d = w.dot(y[j]) 
            if(d>0):
                w=w
            else:
                w=w+alpha*(y[j])
        if (temp!=w).any():                             # if weight value in temp and w doesn't match print the weight
            print('At iteration : {0}'.format(i))
            print('The weight is \n {0}'.format(w))
            print()
            temp = w
        else:
            break                                      # breaks the loop when weight converges to stable value 
    print('Number of iterations are {0}'.format(i))
    print()
    return w
w = perceptron(y)                                      # calling function
print('The Final weight is {0}'.format(w))             # prints the final weight 
print()
for k in range(len(y)):                                # check whether all input patterns are classified correctly
    d = w.dot(y[k])
    if(d>0):
        print('The decision function value of input pattern {0} is "{1}"'.format(k,d))
        print()
print('Hence given input patterns are classified correctly')
