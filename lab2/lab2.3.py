import numpy as np
from matplotlib import pyplot as plt

def main():
    x = np.linspace(0, 5, 100)
    f= np.exp(-x) * np.cos(2*np.pi*x)

    plt.plot(x, f, label="f", marker = 'o')
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('plotting $e^{-x} \dot \cos(2 \pi x)$')
    plt.show()


if __name__=='__main__':
    main()

