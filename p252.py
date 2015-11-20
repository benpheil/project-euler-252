import copy
import matplotlib.pyplot as plt
import numpy as np
import sys
import math

"""
Tried:
    97589.0
"""

def TGenerator(kmax):
    seed = 290797
    idx = 0
    while idx <= kmax:
        idx += 1
        seed = (seed ** 2) % 50515093
        yield (seed % 2000) - 1000

def pointGenerator(kmax):
    T = TGenerator(2*kmax)
    for T1 in T:
        T2 = next(T)
        yield [T1, T2]

class GraphPlotter(object):
    def update(self, points, rootPoint, candidatePoints, poly):
        from matplotlib.patches import Polygon
        x = points[:,0]
        y = points[:,1]
        plt.scatter(x, y, c='b')
        x = candidatePoints[:, 0]
        y = candidatePoints[:, 1]
        plt.scatter(x, y, c='r')
        plt.scatter(rootPoint[0], rootPoint[1], c='g')
        if poly is not None:
            p = Polygon(poly, facecolor='none')
            plt.gca().add_patch(p)
        plt.grid()
        plt.show()

def polygonArea(p):
    # See https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    n = len(p)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += p[i][0] * p[j][1]
        area -= p[j][0] * p[i][1]
    area = abs(area) / 2.0
    return area

def angleBetweenPoints(p, q):
    diff = q - p
    return math.atan2(diff[1], diff[0])

def plot(points, rootPoint, candidatePoints, poly):
    plotter = GraphPlotter()
    plotter.update(points, rootPoint, candidatePoints, poly)

def test():
    # First 3 points are given in problem.
    p = pointGenerator(3)
    assert(next(p) == [527, 144])
    assert(next(p) == [-488, 732])
    assert(next(p) == [-454, -947])

def main():
    kmax = 200
    points = np.array(list(pointGenerator(kmax)))

    for p in points:
        clockwisePoints = sorted(list(points), key=lambda q: angleBetweenPoints(p, q))
        star = [p]
        for q in clockwisePoints:
            if q[0] > p[0]:
                star.append(q)

    plot(points, p, np.array(star), star)

if __name__ == '__main__':
    test()
    main()
