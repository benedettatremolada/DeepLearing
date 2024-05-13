import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize

def true_function(x):
    return np.cos(1.5 * np.pi * x)
def poly1(x, a, b):
    return  a * x + b
def poly4(x, a, b, c, d, e):
    return  a * x**4 + b * x**3 + c * x**2 + d * x + e
def poly15(x, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15):
    return  a15*x**15+a14*x**14+a13*x**13+a12*x**12+a11*x**11+a10*x**10+a9*x**9+a8*x**8 + a7*x**7+a6*x**6+a5*x**5+a4*x**4+a3*x**3+a2*x**2+a1*x+a0


def main():

    x = np.random.rand(30)
    x=np.sort(x)
    x_for_plot=np.linspace(0,1,100)
    y = true_function(x) + np.random.rand() * 0.1

    plt.scatter(x,y, marker='o', label='"true" data', color='black')
    popt, pcov = curve_fit(poly1, x, y)
    plt.plot(x_for_plot,poly1(x_for_plot,*popt), label='Linear')

    popt, pcov = curve_fit(poly4, x, y)
    plt.plot(x_for_plot,poly4(x_for_plot,*popt), label='4 degree polynomial')

    popt, pcov = curve_fit(poly15, x, y)
    plt.plot(x_for_plot,poly15(x_for_plot,*popt), label='15 degree polynomial')

    plt.plot(x_for_plot, true_function(x_for_plot), label="True function")

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