# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 15:43:55 2021

@author: User
"""

import numpy as np

#----------------------------------------
        numpy array 생성
#----------------------------------------
# 1
np.array([1,1,1]) # shape : (3,)

np.array([[1,1,1],[1,1,1]]) # shape : (2, 3) 

np.array([[[1,1,1], [1,1,1]],[[2,2,2],[2,2,2]]]) # shape : (2, 2, 3) = (dim, row, col)

# 2
np.random.randn(3) # -1 ~ 1

np.random.rand(3) # 0 ~ 1

# 3

np.random.uniform(-1, 1, (2,2))
np.random.uniform(0, 1, (2,2))

#----------------------------------------
        numpy 연산자
#----------------------------------------

# 1. np.dot & np.matmul

x = np.random.randn(2)
w1 = np.random.randn(2,3)

np.dot(x, w1)
np.dot(w1, x)       # ValueError : shapes w1 and x not aligned.

np.matmul(x, w1)
np.matmul(w1, x)    # ValueError : 
                    # matmul: Input operand 1 has a mismatch in its core dimension 0, 
                    # with gufunc signature (n?,k),(k,m?)->(n?,m?)

w2 = np.random.randn(2,2)

np.dot(x, w2)
np.dot(w2, x)

np.matmul(x, w2)
np.matmul(w2, x)
''' ---> error 발생하지는 않지만 결과는 달라진다. '''

#----------------------------------------
    np.array 의 deep copy
#----------------------------------------
n1 = np.random.randn(3)
n2 = n1.copy()

n1
n2

n2[0] = 100

n1
n2


#----------------------------------------
    np의 ployfit, poly1d
#----------------------------------------
x = [1,2,3,4]
y = [2,4,7,12]

np.polyfit(x,y,1) # x와 y 사이의 식을 1차원 식으로 구해줌. 첫 요소가 기울기 두번째요소가 절편이다.
np.polyfit(x,y,2) # 2차원 식으로 구해줌.

f1 = np.poly1d(np.polyfit(x,y,1)) # polyfit을 통해 구한 절편과 기울기를 사용해서 함수를 만듬
f1(1) # 1을 입력값을로 했을때 결과를 return해준다.



#----------------------------------------
    np.array의 split, merge, concatenate
#----------------------------------------


