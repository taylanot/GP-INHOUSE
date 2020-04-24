import numpy as np
def fe(x):
    return (6.0*x-2.)**2*np.sin(12*x-4)
def fc(x):
    return np.sin(12*x-4)
def RBF(hyper,x):
    r = np.expand_dims(x/hyper[1],1)-np.expand_dims(x/hyper[1],0)
    diff = np.sum(r**2,axis=2)
    K = hyper[0]* np.exp(-0.5*diff)+np.eye(x.shape[0])*hyper[2]
    dK = np.array([[np.zeros((4,4))],[np.ones((4,4))],[np.ones((4,4))*2]])
    dK[0] = np.exp(-0.5*diff)
    dK[1] = K * diff/ hyper[1]
    dK[2] = np.eye(4)
    return K,dK
def NLML(hyper):
    #print hyper
    lowy_  = np.linspace(0,1,3)
    xe = np.array([0,0.4,0.6,1]).reshape(-1,1)
    n = xe.shape[0]
    ye = fe(xe)
    yt = fc(xe)
    y = (ye-hyper[3]*yt)
    K,dK= RBF(hyper,xe)
    Kinv = np.linalg.inv(K)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T,np.linalg.solve(L,y))
    alphas =np.matmul(alpha,alpha.T)
    ltrace = np.sum(np.log(np.diag(L)))
    obj = -0.5*n*np.log(hyper[0]) - ltrace - 0.5*np.matmul(y.T,alpha)/hyper[0]
    dobj = np.zeros(4)
    dobj[0] = 0.5*np.trace(np.dot((alphas-Kinv),dK[0,0]))/hyper[0] - 0.5*n/hyper[0] + 0.5*np.matmul(y.T,alpha)/hyper[0]/hyper[0]
    dobj[1] = 0.5*np.trace(np.dot((alphas-Kinv),dK[1,0]))/hyper[0]
    dobj[2] = 0.5*np.trace(np.dot((alphas-Kinv),dK[2,0]))/hyper[0]
    dobj[3] = 1/hyper[0]*np.dot((yt).T,alpha)
    return obj,dobj
def centraldiff(id=0,h=0.1):
    hyper1 = np.array([1.,1.,1.,1.])
    hyper2 = np.array([1.,1.,1.,1.])
    hyper1[id] += h
    hyper2[id] -= h
    res1,_ = NLML(hyper1);
    res2,_ = NLML(hyper2)
    return (res1-res2)/(2*h)
xe = np.array([0,0.4,0.6,1]).reshape(-1,1)
hyper = np.array([1.,1.,1.,1.])

check = centraldiff(id=3,h=0.0001)
print check
print (NLML(hyper))
