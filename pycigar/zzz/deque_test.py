import numpy as np

from collections import deque

x = deque([1, 2, 3], maxlen=3)

print(x)

x.append(4)

print(x)

x = deque([0], maxlen=2)

print(x)

x.append(50)

print(x)

x.append(60)

print(x)

print(x[0], x[1])

y = [1, 2]
print(y)

print(y[0])

x = deque([], maxlen=2)
x.append('charge')

print(x)

x.append('charge')

print(x)

x.append('discharge')

print(x)

# print(x[0:2])
