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

class DelaunayWrapper(object):
    def __init__(self, points):
        self._delaunay = scipy.spatial.Delaunay(points)

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

        return zip(simplex_indices, self._triangles)

    def pointIndiciesInTriangle(self, idx):
        return self._delaunay.simplices[idx][:]

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

    points = np.array(list(pointGenerator(kmax)))

    d = DelaunayWrapper(points)

    trianglesSorted = sorted(d.triangles, key=lambda t: triangleArea(t[1]))
    for idx, triangle in trianglesSorted:
        print("{0} ({1}): {2}".format(idx, triangleArea(triangle), triangle))

    biggestIdx = trianglesSorted[-1][0]
    print(biggestIdx)
    biggestIndicies = d.pointIndiciesInTriangle(biggestIdx)
    poly = points[biggestIndicies]
    print(biggestIndicies)

    plot(points, poly)

if __name__ == '__main__':
    test()
    main()
