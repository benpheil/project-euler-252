import copy
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
import sys

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
    def update(self, points, poly):
        from matplotlib.patches import Polygon
        x = points[:,0]
        y = points[:,1]
        plt.scatter(x, y)
        if poly is not None:
            p = Polygon(poly)
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

def plot(points, poly):
    plotter = GraphPlotter()
    plotter.update(points, poly)

def test():
    # First 3 points are given in problem.
    p = pointGenerator(3)
    assert(next(p) == [527, 144])
    assert(next(p) == [-488, 732])
    assert(next(p) == [-454, -947])

def main():
    kmax = 500
    points = np.array(list(pointGenerator(kmax)))

    plot(points, None)

if __name__ == '__main__':
    test()
    main()
