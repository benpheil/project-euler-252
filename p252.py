import copy
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
import sys

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
        p = Polygon(poly)
        x = points[:,0]
        y = points[:,1]
        plt.scatter(x, y)
        plt.gca().add_patch(p)
        plt.show()

def triangleArea(t):
    # See http://geomalgorithms.com/a01-_area.html
    return abs(((t[1][0] - t[0][0]) * (t[2][1] - t[0][1]) - (t[2][0] - t[0][0]) * (t[1][1] - t[0][1])) / 2.0)

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

class DelaunayWrapper(object):
    def __init__(self, points):
        self._delaunay = scipy.spatial.Delaunay(points, incremental=True)

    @property
    def adjacency(self):
        if not hasattr(self, '_adjacency'):
            v = self._delaunay.vertex_neighbor_vertices
            self._adjacency = list()
            indices = v[0]
            indptr = v[1]

            for idx in range(0, len(v[0]) - 1):
                for neighbor in indptr[indices[idx]:indices[idx+1]]:
                    self._adjacency.append([idx, neighbor])

            self._adjacency = np.array(self._adjacency)

        return self._adjacency

    @property
    def triangles(self):
        if not hasattr(self, '_triangles'):
            num_simplices = self._delaunay.simplices.shape[0]
            simplex_indices = list(range(0, num_simplices))
            self._triangles = list()
            for simplex_idx in simplex_indices:
                simplex = self._delaunay.simplices[simplex_idx, :]
                self._triangles.append([self._delaunay.points[simplex[0]],
                                        self._delaunay.points[simplex[1]],
                                        self._delaunay.points[simplex[2]],
                                       ])

        return [Triangle(idx, points) for idx, points in zip(simplex_indices, self._triangles)]

    def pointIndiciesInTriangle(self, triangleIndex):
        return self._delaunay.simplices[triangleIndex][:]

    def addPoints(self, points):
        self._delaunay.add_points(points)

class Triangle(object):
    """ A triangle derived from a DelaunayWrapper. """
    def __init__(self, triangleIndex, points):
        self.triangleIndex = triangleIndex
        self.points = points
        self.area = triangleArea(points)

class Polygon(object):
    """ A polygon comprised of Triangles. """
    def __init__(self, parent=None):
        if parent is not None:
            self = copy.copy(parent)
        else:
            self.delaunay = None
            self.triangles = set()
            self._points = set()

    def addTriangle(self, triangle):
        self.triangles.add(triangle)
        for p in triangle.points:
            self._points.add(tuple(p))
        if self.delaunay is not None:
            self.delaunay.addPoints(triangle.points)
        elif len(list(self.triangles)) > 1:
            self.delaunay = DelaunayWrapper(list(self._points))
        assert(np.array_equal(set(tuple(p) for p in self.points), self._points))

    @property
    def points(self):
        hull =  scipy.spatial.ConvexHull(list(self._points))
        return hull.points[hull.vertices]

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
    kmax = 20

    # Generate the points.
    points = np.array(list(pointGenerator(kmax)))

    # Calculate the Delaunay triangulation of the points.
    d = DelaunayWrapper(points)

    # Find the largest triangle.
    trianglesSorted = sorted(d.triangles, key=lambda t: t.area)

    # Initialize the candidate polygon with it
    poly = Polygon()
    poly.addTriangle(trianglesSorted[-1])

    poly.addTriangle(trianglesSorted[-2])

    print("Points: {0}".format(poly.points))

    plot(points, poly.points)

if __name__ == '__main__':
    test()
    main()
