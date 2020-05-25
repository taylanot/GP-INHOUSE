import numpy as np
# Created a postive definite matrix then checked the mathc of Perdikaris2016,
# but got matching results only for the case sn=0.
# Thus, my conclusion is that their math is inconsistant/incorrect and I am using my own
# formulation.
a = np.random.uniform(1,3,(3,3))
sf = 5.
sn = 2.1
K = np.dot(a,a.T)
R = sf*K + np.eye(3) * sn

print "K:\n", K

print "R=sf*K+sn*I:\n ", R


print "R^(-1):\n", np.linalg.inv(R)

print "(K+sn*I)^(-1):\n", 1./sf*np.linalg.inv(K + np.eye(3) * sn)

print "log|R|: ", np.log(np.linalg.det(R))

print "nlog(sf)+log|K+sf*I|: ", 3*np.log(sf) + np.log(np.linalg.det(K + np.eye(3) * sn))



