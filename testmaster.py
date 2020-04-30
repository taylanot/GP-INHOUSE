# (c) Ozgur Taylan TURAN / Delft University of Technology / 2020, March
################################################################################
# TESTING IMPLIMENTATIONS of Multifidelity Gaussian Process Regression,
# Method:Perdikaris 2016/ Test: Forrester 2007, One variable demonstration
################################################################################
# Import the necessary modules
################################################################################
import sys;sys.dont_write_bytecode = True;
from gpmaster import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
################################################################################
np.random.seed(10);
# Expensive Function
def fe(x):
    return (6.0*x-2.)**2*np.sin(12*x-4)
# Cheap Function
def fc(x):
    A = 0.5; B=10; C=5
    return A*fe(x) +B*(x-0.5)-C
################################################################################
plt.figure(1)
plt.subplot(121)
Xe = np.array([0,0.4,0.6,1]).reshape(-1,1)
plt.scatter(Xe,fe(Xe),color='k',label='Expensive Sample')
Xc = np.linspace(0,1,11).reshape(-1,1)
plt.scatter(Xc,fc(Xc),marker='^',color='k',label='Cheap Sample')
x  = np.linspace(0,1,100).reshape(-1,1)
plt.plot(x,fe(x),'-',color='k',linewidth=2.,label='Expensive')
plt.plot(x,fc(x),'-',color='darkorange',label='Cheap')
# Regular GPR with just Expnesive sample
reg1 = GPR(Xe,fe(Xe),noise_var=None,noise_fix=False);
reg1.plot(name='Expensive',plot_std=False)
# Regular GPR with just Expnesive sample
reg2 = multiGPR(Xc,Xe,fc(Xc),fe(Xe));
reg2.plot(name='Expensive',plot_std=True)
print reg2.params
m2,std2 = reg2.inference(x,return_std=True)
reg2.getParams()
plt.legend()
# Error Calculation
print('multiGPR ||Error|| = {}'.format(np.linalg.norm(std2)))
plt.subplot(122)
plt.plot(x,std2**2,color='k',label='variance')
plt.xlabel('$x$')
plt.ylabel('$variance$')
plt.legend()
#print(reg2.K)
plt.show()
