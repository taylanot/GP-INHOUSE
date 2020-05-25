import numpy as np 
from gp import *
import matplotlib.pyplot as plt
def elastic(eps):
    De = 3760
    return De*eps
def felastic(eps):
    De = 1.20535955e+04 
    return De*eps
fdata = np.loadtxt("lodi.dat")
fstrain = fdata[:,1]
fstress = fdata[:,2] - felastic(fstrain)
data = np.loadtxt("lodi-homo.dat")
strain = data[:,1]
stress = data[:,2] - elastic(strain)
strainlast = fstrain[-1]
stresslast = fstress[-1]
plt.figure(0)
plt.plot(strain,stress,label="homo")
plt.plot(fstrain,fstress,label="hetero")
#plt.plot(strain,elastic(strain),label="Elastic")
#plt.plot(strain,felastic(strain),label="fElastic")
plt.legend()
plt.xlabel("Strain")
plt.ylabel("Stress")
plt.figure(1)
plt.plot(strain,stress,label="homo")
plt.plot(fstrain,fstress,label="hetero")
#plt.plot(strain,elastic(strain),label="Elastic")
#plt.plot(strain,felastic(strain),label="fElastic")
plt.legend()
plt.xlabel("Strain")
plt.ylabel("Stress")
selectlow = np.arange(0,100,5)
select = np.arange(0,50,10)
stress = stress[selectlow].reshape(-1,1)
strain = strain[selectlow].reshape(-1,1)
fstrain = fstrain[select].reshape(-1,1)
fstress = fstress[select].reshape(-1,1)
#fstrain = np.vstack((fstrain,strainlast))
#fstress = np.vstack((fstress,stresslast))
plt.figure(0)
reg1 = GPR(fstrain,fstress)
reg1.plot("hetero-reg",plot_std=True)
plt.figure(1)
reg2 = multiGPR(strain,fstrain,stress,fstress)
reg3  = GPR(strain,stress)
reg3.plot("homo-reg",plot_std=True)
reg2.plot("hetero-multireg",plot_std=True)
plt.legend()
plt.show()


