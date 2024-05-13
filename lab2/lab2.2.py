import numpy as np
import time
import numba as nb
from numpy import random



def basic_dot(A,v):
    N=v.size
    ans=np.zeros(N)
    
    for i in range(N):
        for j in range(N):
            ans[i]+=A[i][j]*v[j]
    return ans

@nb.njit(parallel=True)
def numba_dot(A,v):
    N=v.size
    ans=np.zeros(N)
    
    for i in range(N):
        for j in range(N):
            ans[i]+=A[i][j]*v[j]
    return ans


def main():
    N = int(input('Enter N: '))
    #v= random.randint(10, size=(N)) random int among 0-10
    v = np.random.rand(N).astype(np.float64)
    A = np.random.rand(N, N).astype(np.float64)
    
    start_t=time.time()
    res1=basic_dot(A,v)
    print(f'Execution time, basic_dot: {time.time()-start_t}')

    start_t=time.time()
    res2=A.dot(v)
    print(f'Execution time, numpy dot: {time.time()-start_t}')

    start_t=time.time()
    res=numba_dot(A,v)
    print(f'Execution time, numba dot (second run): {time.time()-start_t}')


if __name__=='__main__':
    main()
