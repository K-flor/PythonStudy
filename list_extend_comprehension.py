
"""
list의 extend 함수.

왜 list comprehension이 적용되지 않을까?
"""

a = [[1,2,3], ['a', 'b', 'c', 'd'],[1.02, 5.11, 12.2]]

''' for loop 를 사용하여 extend 하는 경우'''
b1 = []
for i in a:
    b1.extend(i)

# OUTPUT
# [1, 2, 3, 'a', 'b', 'c', 'd', 1.02, 5.11, 12.2]


''' list comprehension을 사용하여 extend 하는 경우 '''

b2 = []
b2.extend(i for i in a)

# OUTPUT
# [[1, 2, 3], ['a', 'b', 'c', 'd'], [1.02, 5.11, 12.2]]