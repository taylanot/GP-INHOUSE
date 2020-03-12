# (c) Ozgur Taylan TURAN / Delft University of Technology / 2020, March
################################################################################
# TESTING IMPLIMENTATIONS of Multifidelity Gaussian Process Regression,
# Method:Perdikaris 2016/ Test: Ghoreishi2018 One variable demonstration
################################################################################
# Import the necessary modules
################################################################################
import sys;sys.dont_write_bytecode = True;
from gp import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
################################################################################
# FUNCTIONS
################################################################################
def f1(x):
    return -5*(2 - (1.8-3*x) * np.sin(18*x+0.1)) - 3*x**2
def f2(x):
    return 2 - (1.8-3*x) * np.sin(18*x)
def f3(x):
    return 2*(2 - (1.6-3*x) * np.sin(18*x))
################################################################################
Xc = np.linspace(0,1.2,10).reshape(-1,1)
Xe = np.linspace(0,1.2,4).reshape(-1,1)
plt.scatter(Xc,f2(Xc),color='k')
plt.scatter(Xe,f1(Xe),color='k',marker='^')
x = np.linspace(0,1.2,100)
plt.plot(x,f1(x),label='f1(x)',color='k')
#plt.plot(x,f2(x),label='f2(x)',color='b')
plt.plot(x,f2(x),label='f2(x)',color='r')
reg = multiGPR(Xc,Xe,f2(Xc),f1(Xe))
reg.plot(name='Reg',plot_std=True)
reg1 = GPR(Xe,f1(Xe))
reg1.plot(name='Reg',plot_std=True)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.show()
