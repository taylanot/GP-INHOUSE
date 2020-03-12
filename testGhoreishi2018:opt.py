# (c) Ozgur Taylan TURAN / Delft University of Technology / 2020, March
################################################################################
# TESTING IMPLIMENTATIONS of Multifidelity Gaussian Process Regression,
# Method:Perdikaris 2016/ Test: Forrester 2007, One variable demonstration
################################################################################
# Import the necessary modules
################################################################################
import sys;sys.dont_write_bytecode = True;
from gp import *
from opt import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
################################################################################
################################################################################
# FUNCTIONS
################################################################################
def f1(x):
    return -5*(2 - (1.8-3*x) * np.sin(18*x+0.1)) - x**2
def f2(x):
    return 2*(2 - (1.6-3*x) * np.sin(18*x))
def f3(x):
    return 2 - (1.4-3*x) * np.sin(18*x)
################################################################################
Xc = np.linspace(0,1.2,10).reshape(-1,1)
Xe = np.linspace(0,1.2,4).reshape(-1,1)
for iter in range(0,4):
    plt.figure(iter)
    plt.subplot(211)
    plt.scatter(Xc,f3(Xc),color='k')
    plt.scatter(Xe,f1(Xe),color='k',marker='^')
    x = np.linspace(0,1.2,100).reshape(-1,1)
    plt.plot(x,f1(x),label='f1(x)',color='k')
    #plt.plot(x,f2(x),label='f2(x)',color='b')
    plt.plot(x,f3(x),label='f3(x)',color='r')
    reg = multiGPR(Xc,Xe,f3(Xc),f1(Xe))
    reg.plot(name='Expensive')
    low = GPR(Xe,f1(Xe));low.plot(name='Expensive')
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.subplot(212)
    EI = BO(reg,Xe,f1(Xe),x).Expected_Improvement()
    Xnew = BO(reg,Xe,f1(Xe),x).prop()
    Xe = np.vstack([Xe,Xnew])
    plt.plot(x,EI,color='k',label='EI')
    plt.xlabel("$x$")
    plt.ylabel("$EI(x)$")
    plt.legend()
plt.show()
