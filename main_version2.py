import numpy as np
import matplotlib.pyplot as plt
import cProfile
import pstats

from numba import njit         # Added in v2

@njit                          # Added in v2
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
    grid = []
    for y in y_space:
        row = []
        for x in x_space:
            row.append(x + y * 1j)
        grid.append(row)
    return grid

def main():
    grid = complexMatrix(-3, 1, -2, 2, 0.001)
 
    isConvergent = []
    for row in grid:
        result_row = []
        for point in row:
            result_row.append(checkConvergence(point))
        isConvergent.append(result_row)
 
    plt.imshow(isConvergent, cmap='hot')
    plt.show()
 
if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename="v2.prof")
    # python -m snakeviz v2.prof


