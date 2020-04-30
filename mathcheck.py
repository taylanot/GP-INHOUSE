import numpy as np

a = np.random.uniform(1,3,(3,3))
sf = 5.
sn = 0.1
K = np.dot(a,a.T)
R = sf*K + np.eye(3) * sn
print np.linalg.inv(R)

print 1./sf*np.linalg.inv(K + np.eye(3) * sn)


print np.log(np.linalg.det(R))

print 3*np.log(sf) + np.log(np.linalg.det(K + np.eye(3) * sn))



