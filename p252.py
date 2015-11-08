import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
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
    def __init__(self):
        # Enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        self._w = pg.GraphicsWindow()
        self._w.setWindowTitle('Problem 252 Graph')
        self._v = self._w.addViewBox()
        self._v.setAspectLocked()

        self._g = pg.GraphItem()
        self._v.addItem(self._g)

    def update(self, points, adjacency):
        symbols = ['o'] * len(points)
        self._g.setData(pos=points, adj=adjacency, size=10, symbols=symbols, pxMode=False)

def adjacencyFromDelaunay(d):
    v = d.vertex_neighbor_vertices
    adjacency = list()
    indices = v[0]
    indptr = v[1]

    for idx in range(0, len(v[0]) - 1):
        for neighbor in indptr[indices[idx]:indices[idx+1]]:
            adjacency.append([idx, neighbor])

    adjacency = np.array(adjacency)

    return adjacency

def plot(points, adjacency):
    plotter = GraphPlotter()
    plotter.update(points, adjacency)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


def test():
    # First 3 points are given in problem.
    p = pointGenerator(3)
    assert(next(p) == [527, 144])
    assert(next(p) == [-488, 732])
    assert(next(p) == [-454, -947])

def main():
    kmax = 20

    points = np.array(list(pointGenerator(kmax)))

    d = scipy.spatial.Delaunay(points)

    adjacency = adjacencyFromDelaunay(d)

    plot(points, adjacency)

if __name__ == '__main__':
    test()
    main()
