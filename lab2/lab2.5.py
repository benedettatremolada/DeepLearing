import numpy as np
from matplotlib import pyplot as plt

def main():
    x = np.linspace(-3, 3, 100)
    f = -np.sin(x*x)/x + 0.01 * x*x

    np.savetxt("output.dat", np.vstack([x,f]))


    plt.plot(x, f, label="f", marker = 'o')
    plt.grid()
    plt.xlabel('x')
    plt.xlim=(-3,3)
    plt.ylabel('y')
    plt.title('plotting $e^{-x} \dot \cos(2 \pi x)$')
    plt.show()

    plt.savefig('output5.png')

if __name__=='__main__':
    main()