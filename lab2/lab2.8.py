import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import  minimize

def true_function(x):
    return np.cos(1.5 * np.pi * x)
    

def main():
    x=np.random.rand(30)
    x=np.sort(x)
    x_for_plot=np.linspace(0,1,100)
    y=true_function(x)+np.random.rand(30)*0.1

    def loss(p, func):
        ypred = func(list(p),x)
        return tf.reduce_mean(tf.square(ypred - y)).numpy()

    for degree in [1, 4, 15]:
        res = minimize(loss, np.zeros(degree+1), args=(tf.math.polyval), method='BFGS')
        plt.plot(x_for_plot, np.poly1d(res.x)(x_for_plot), label=f"Poly degree={degree}")

    plt.plot(x_for_plot, true_function(x_for_plot), label="True function")
    plt.scatter(x,y, marker='o', label='data', color='black')
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([0,1])
    plt.ylim([-1,1])
    plt.title('Fitting my function')
    plt.legend()
    plt.show()

if __name__=='__main__':
    main()