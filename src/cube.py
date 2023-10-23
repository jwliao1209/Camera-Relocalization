import numpy as np


class Point:
    def __init__(self, coordinate: np.array, color: tuple):
        self.coordinate = coordinate.reshape(1, 3)
        self.color = color


def get_cube_points(n):
    x1 = np.linspace(0, 1, n)
    x2 = np.linspace(0, 1, n)
    X1, X2 = np.meshgrid(x1, x2)

    point_list = []

    ## up
    for x1, x2 in np.nditer([X1, X2]):
        coor = np.array([x1, x2, 1])
        color = (255, 0, 0)
        point_list.append(Point(coor, color))

    ## down
    for x1, x2 in np.nditer([X1, X2]):
        coor = np.array([x1, x2, 0])
        color = (0, 255, 0)
        point_list.append(Point(coor, color))

    ## left
    for x1, x2 in np.nditer([X1, X2]):
        coor = np.array([0, x1, x2])
        color = (0, 0, 255)
        point_list.append(Point(coor, color))

    ## right
    for x1, x2 in np.nditer([X1, X2]):
        coor = np.array([1, x1, x2])
        color = (255, 255, 0)
        point_list.append(Point(coor, color))
        
    ## front
    for x1, x2 in np.nditer([X1, X2]):
        coor = np.array([x1, 0, x2])
        color = (255, 0, 255)
        point_list.append(Point(coor, color))

    ## back
    for x1, x2 in np.nditer([X1, X2]):
        coor = np.array([x1, 1, x2])
        color = (0, 255, 255)
        point_list.append(Point(coor, color))

    return point_list
