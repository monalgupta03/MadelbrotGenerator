import numpy as np
import matplotlib.pyplot as plt
import cProfile
import pstats

from numba import njit

@njit 
def checkConvergence(number : "complex", z : "complex" = 0) -> bool:
    iterNo = 0
    while (abs(z) < 1000) and (iterNo < 100):
        z = z*z + number
        iterNo+= 1
    if abs(z) >= 1000:
        return False
    else:
        return True


def complexMatrix(x_start: float, x_end: float, y_start: float, y_end: float, stepSize: float):
    x_space = np.linspace(x_start, x_end, int((x_end - x_start)/stepSize) + 1)
    y_space = np.linspace(y_start, y_end, int((y_end - y_start)/stepSize) + 1)
    grid = 1j*y_space[:, np.newaxis] + x_space[np.newaxis, :]
    return grid


#Added in v4
def getConvergentMatrix(grid):
    return np.zeros((len(grid), len(grid[0])),dtype=bool)

#Added in v4
@njit
def iterateOverConvergentMatrix(grid, isConvergent):
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            isConvergent[i][j] = checkConvergence(grid[i][j])

#Added in v4
def complexMatrixWrapper(grid):
    isConvergent = getConvergentMatrix(grid)
    iterateOverConvergentMatrix(grid, isConvergent)
    return isConvergent


def main():
    grid = complexMatrix(-3, 1, -2, 2, 0.001)
    isConvergent = complexMatrixWrapper(grid)     #Added in v4
    plt.imshow(isConvergent, cmap='hot')
    plt.show()
 
if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename="v4.prof")
    # python -m snakeviz v4.prof


