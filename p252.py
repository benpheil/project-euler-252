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
        plt.grid()
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
        self._triangles = list()

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
    def _num_simplices(self):
        return self._delaunay.simplices.shape[0]

    @property
    def triangles(self):
        simplex_indices = list(range(0, self._num_simplices))
        for simplex_idx in simplex_indices:
            simplex = self._delaunay.simplices[simplex_idx, :]
            self._triangles.append([self._delaunay.points[simplex[0]],
                                    self._delaunay.points[simplex[1]],
                                    self._delaunay.points[simplex[2]],
                                   ])

        return [Triangle(idx, points) for idx, points in zip(simplex_indices, self._triangles)]

    def pointIndiciesInTriangle(self, triangleIndex):
        return self._delaunay.simplices[triangleIndex][:]

    def addTriangle(self, triangle):
        self._delaunay.add_points(triangle.points)
        self._triangles.append(triangle)

    def neighbors(self, triangle):
        neighborIndices = list()
        for simplexIdx in self._delaunay.neighbors[triangle.triangleIndex, :]:
            if simplexIdx != -1:
                neighborIndices.append(simplexIdx)

        neighborTriangles = list()
        for triangle in self.triangles:
            if triangle.triangleIndex in neighborIndices:
                neighborTriangles.append(triangle)

        return neighborTriangles

class Triangle(object):
    """ A triangle derived from a DelaunayWrapper. """
    def __init__(self, triangleIndex, points):
        self.triangleIndex = triangleIndex
        self.points = points
        self.area = triangleArea(points)

    def __str__(self):
        return "[{0}: {1}, {2}, {3}, A = {4}]".format(
                self.triangleIndex,
                self.points[0],
                self.points[1],
                self.points[2],
                self.area)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return self.triangleIndex

    def __eq__(self, other):
        return self.triangleIndex == other.triangleIndex

class Polygon(object):
    """ A polygon comprised of Triangles. """
    def __init__(self, parent=None):
        if parent is not None:
            self = copy.copy(parent)
        else:
            self.delaunay = None
            self.triangles = set()
            self._points = set()

    def __str__(self):
        return ", ".join([str(p) for p in self._points])

    def addTriangle(self, triangle):
        """ Merge a triangle into the Polygon.  Asserts that the resulting polygon is convex. """
        assert(triangle not in self.triangles)
        self.triangles.add(triangle)
        for p in triangle.points:
            self._points.add(tuple(p))
        if self.delaunay is not None:
            self.delaunay.addTriangle(triangle)
        elif len(list(self.triangles)) > 1:
            self.delaunay = DelaunayWrapper(list(self._points))
        assert(np.array_equal(set(tuple(p) for p in self.points), self._points))

    def neighbors(self, d):
        """ Get the Triangles in DelaunayWrapper `d` adjacent to the Polygon. """
        _neighbors = list()
        for triangle in self.triangles:
            for neighbor in d.neighbors(triangle):
                if neighbor not in self.triangles:
                    _neighbors.append(neighbor)

        return _neighbors

    @property
    def points(self):
        """ Return the Polygon's points in counterclockwise order. """
        hull =  scipy.spatial.ConvexHull(list(self._points))
        return hull.points[hull.vertices]

    def isConvexWithTriangle(self, triangle):
        points = copy.copy(self._points)
        for p in triangle.points:
            points.add(tuple(p))
        hull = scipy.spatial.ConvexHull(list(points))
        hullPts = set()
        for p in hull.points[hull.vertices]:
            hullPts.add(tuple(p))
        return points == hullPts

    @property
    def area(self):
        return sum(t.area for t in self.triangles)

def plot(points, poly):
    plotter = GraphPlotter()
    plotter.update(points, poly)

def solve(kmax):
    # Generate the points.
    points = np.array(list(pointGenerator(kmax)))

    # Calculate the Delaunay triangulation of the points.
    d = DelaunayWrapper(points)

    # Find the largest triangle.
    trianglesSorted = sorted(d.triangles, key=lambda t: t.area)

    # Initialize the candidate polygon with it
    poly = Polygon()
    poly.addTriangle(trianglesSorted[-1])

    while True:
        aMax = 0.
        biggest = None
        for neighbor in poly.neighbors(d):
            if neighbor.area > aMax and poly.isConvexWithTriangle(neighbor):
                aMax = neighbor.area
                biggest = neighbor

        if biggest is None:
            # We can't add any triangles and maintain convexity.
            break;
        else:
            poly.addTriangle(biggest)

    return poly

def test():
    # First 3 points are given in problem.
    p = pointGenerator(3)
    assert(next(p) == [527, 144])
    assert(next(p) == [-488, 732])
    assert(next(p) == [-454, -947])

    poly = solve(20)
    assert(poly.area == 1049694.5)

def main():
    kmax = 20
    points = np.array(list(pointGenerator(kmax)))
    poly = solve(kmax)
    print("Polygon: {0}".format(poly))
    print("Area: {}".format(poly.area))
    plot(points, poly.points)

if __name__ == '__main__':
    test()
    main()
