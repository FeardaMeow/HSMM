from numba import jit
import numpy as np
import time

@jit(nopython = True)
def go_fast(x):
    data_list = []
    for i in range(x.shape[0]):
        data_list.append(x[i,:])
    return data_list

def main():
    x = np.arange(300).reshape((100,-1))

    start = time.time()
    data_list = go_fast(x)
    end = time.time()
    print("Elapsed (with compilation) = %s" % (end - start))

    start = time.time()
    data_list = go_fast(x)
    end = time.time()
    print("Elapsed (without compilation) = %s" % (end - start))

if __name__ == "__main__":
    main()