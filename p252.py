import matplotlib.pyplot as plt
import numpy
import scipy.spatial
import itertools

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
        yield (T1, T2)

class GraphPlotter(object):
    def update(self, points, poly):
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        plt.scatter(x, y)
        if poly is not None:
            from matplotlib.patches import Polygon
            hull = scipy.spatial.ConvexHull(list(p for p in poly))
            p = Polygon(hull.points[hull.vertices])
            plt.gca().add_patch(p)
        plt.grid()

def plot(points, poly):
    plotter = GraphPlotter()
    plotter.update(points, poly)

def subsets(S, r):
    """ Return the set of subsets of set S, which have cardinality r. """
    return set(map(frozenset, itertools.combinations(S, r)))

def convexHull(polygon):
    return scipy.spatial.Delaunay(list(p for p in polygon))

def inHull(points, delaunay):
    _points = numpy.array([p for p in points])
    return numpy.sum(delaunay.find_simplex(_points) >= 0) > delaunay.points.shape[0]

def area(points):
    arr = numpy.array(points)
    x = arr[:,0]
    y = arr[:,1]
    return 0.5*numpy.abs(numpy.dot(x,numpy.roll(y,1))-numpy.dot(y,numpy.roll(x,1)))

def solve(points, rmax):
    aMax = 0
    biggest = None
    for numVertices in range(3, rmax + 1):
        print("aMax = {}, numVertices = {} of {}".format(aMax, numVertices, rmax))
        for candidatePolygon in subsets(points, numVertices):
            # FIXME: This is really redundant.
            delaunay = scipy.spatial.Delaunay(list(p for p in candidatePolygon))
            hull = scipy.spatial.ConvexHull(list(p for p in candidatePolygon))
            if candidatePolygon == frozenset(map(tuple, hull.points[hull.vertices].tolist())):
                if not inHull(points, delaunay):
                    a = area(hull.points[hull.vertices])
                    if a > aMax:
                        aMax = a
                        biggest = candidatePolygon
    return aMax, biggest

def test():
    # First 3 points are given in problem.
    p = pointGenerator(3)
    assert(next(p) == (527, 144))
    assert(next(p) == (-488, 732))
    assert(next(p) == (-454, -947))

    # The case for kmax = 20 is given in the problem.
    points = set(pointGenerator(20))
    aMax, biggest = solve(points, 7)
    assert(aMax == 1049694.5)

def main():
    # Cardinality of the point set.
    kmax = 20
    # Maximum number of verticies in the polygon.
    rmax = 7

    points = set(pointGenerator(kmax))
    aMax, biggest = solve(points, rmax)

    print("Biggest (A = {}): {}".format(aMax, biggest))
    plot(points, biggest)
    plt.show()

if __name__ == '__main__':
    test()
    main()
