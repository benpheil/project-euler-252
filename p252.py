import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial

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
        x = points[:,0]
        y = points[:,1]
        plt.scatter(x, y)
        if poly is not None:
            from matplotlib.patches import Polygon
            p = Polygon(poly)
            plt.gca().add_patch(p)
        plt.grid()
        plt.show()

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
    plot(points, None)

if __name__ == '__main__':
    test()
    main()
