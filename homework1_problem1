### Homework 1_probability Review No.1 ###

"""
basic approach of the problem
suppose there is square(width 1, length 1), and circle(radius: 1 / 2)
so the size of square is 1, and circle is pi
the ratio of square and circle is pi / 4
generate random point (0 < x < 1, 0 < y < 1)
classify the point whether it is in the circle or out of the circle.
-> calculating the distance from center of the circle

Finally we can get the ratio of size: (number of points in the circle) / (number of points in the square)
The more points are generated, the ratio gets closer to pi / 4
So the estimated pi = 4 * ((number of points in the circle) / (number of points in the square))
This approach is called Monte Carlo Method
"""

import random

# set iteration : after some experiments, found that iterations should be at least 1000000
iter = 1000000
inside_circle = 0
inside_square = 0

for i in range(iter):
    point_x = random.random()
    point_y = random.random()

    # check if  the point is in circle or not
    distance_squared = (point_x - 1/2)**2 + (point_y - 1/2)**2

    if distance_squared > 1 / 4:
        inside_square += 1
    else:
        inside_circle += 1
        inside_square += 1

ratio = inside_circle / inside_square

answer = 4 * ratio
print(answer)
