import copy
import matplotlib.pyplot as plt
import numpy as np
import sys
import math

"""
Tried:
    97589.0
    191815.0
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

def turnDirection(p1, p2, p3):
    if (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) > 0:
        return 'left'
    else:
        return 'right'

def makeStar(p, points):
    ccwPoints = sorted(list(points), key=lambda q: angleBetweenPoints(p, q))
    star = [p]
    for q in ccwPoints:
        if q[0] > p[0]:
            star.append(q)

    return star

def visibilityGraph(star):
    queues = [list() for p in star]
    edges = dict()
    for i in range(0, len(star)):
        edges[i] = set()

    def turn(ij, jk):
        assert(ij[1] == jk[0])
        return turnDirection(star[ij[0]], star[ij[1]], star[jk[1]])

    def proceed(i, j):
        while len(queues[i]) > 0 and turn((queues[i][0], i), (i, j)) == 'left':
            proceed(queues[i][0], j)
            queues[i].pop(0)
        edges[i].add(j)
        edges[j].add(i)
        queues[j].insert(0, i)

    for idx in range(0, len(star) - 1):
        proceed(idx, idx + 1)

    return edges

def solve(points):
    aMax = 0
    biggest = None
    bestStar = None
    bestP = None
    for p in points:
        star = makeStar(p, points)
        if len(star) < 3:
            continue

        # Walk the points ccw
        hole = [star[0], star[1]]
        for idx in range(2, len(star)):
            if turnDirection(hole[-2], hole[-1], star[idx]) == 'left':
                hole.append(star[idx])
        area = polygonArea(hole)
        if area > aMax:
            aMax = area
            biggest = hole
            bestStar = star
            bestP = p

        # Walk the points cw
        hole = [star[-1], star[-2]]
        for idx in reversed(range(0, len(star) - 2)):
            if turnDirection(hole[-2], hole[-1], star[idx]) == 'right':
                hole.append(star[idx])
        area = polygonArea(hole)
        if area > aMax:
            aMax = area
            biggest = hole
            bestStar = star
            bestP = p

    return aMax, biggest, bestStar, bestP

def plot(points, rootPoint, candidatePoints, poly):
    plotter = GraphPlotter()
    plotter.update(points, rootPoint, candidatePoints, poly)

def test():
    # First 3 points are given in problem.
    p = pointGenerator(3)
    assert(next(p) == [527, 144])
    assert(next(p) == [-488, 732])
    assert(next(p) == [-454, -947])

    aMax, biggest, bestStar, bestP = solve(np.array(list(pointGenerator(20))))
    assert(aMax == 1049694.5)

def main():
    kmax = 500
    points = np.array(list(pointGenerator(kmax)))
    
    aMax, biggest, bestStar, bestP = solve(points)

    print("Largest area: {}".format(aMax))
    plot(points, bestP, np.array(bestStar), biggest)

if __name__ == '__main__':
    test()
    main()
