import numpy as np
def fe(x):
    return (6.0*x-2.)**2*np.sin(12*x-4)
def fc(x):
    return np.sin(12*x-4)
def RBF(hyper,x,ubnd_,lbnd_):
    r = np.expand_dims(x/hyper[1],1)-np.expand_dims(x/hyper[1],0)
    diff = np.sum(r**2,axis=2)
    K = hyper[0]* np.exp(-0.5*diff)+np.eye(x.shape[0])*hyper[2]
    dK = np.array([[np.zeros((4,4))],[np.ones((4,4))],[np.eye((4))]])
    dK[0] = hyper[0] * np.exp(-0.5*diff) *(ubnd_[0]-lbnd_[0])
    dK[1] = hyper[0] * diff * np.exp(-0.5*diff) * (ubnd_[1]-lbnd_[1])
    dK[2] *= hyper[2] * (ubnd_[2]-lbnd_[2])
    return K,dK
def NLML(vars,ubnd_,lbnd_):
    hyper = np.zeros(vars.shape[0])
    for i in range(0,3):
        ubnd_[i] = np.log(ubnd_[i]);lbnd_[i] = np.log(lbnd_[i])
        hyper[i] = np.exp(vars[i] * (ubnd_[i]-lbnd_[i]) + lbnd_[i])
    hyper[-1] = vars[-1] * (ubnd_[-1]-lbnd_[-1]) + lbnd_[-1]
    lowy_  = np.linspace(0,1,3)
    xe = np.array([0,0.4,0.6,1]).reshape(-1,1)
    n = xe.shape[0]
    ye = fe(xe)
    yt = fc(xe)
    y = (ye-hyper[3]*yt)
    K,dK= RBF(hyper,xe,ubnd_,lbnd_)
    Kinv = np.linalg.inv(K)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T,np.linalg.solve(L,y))
    alphas =np.matmul(alpha,alpha.T)
    ltrace = np.sum(np.log(np.diag(L)))
    obj = -0.5*n*np.log(hyper[0]) - ltrace - 0.5*np.matmul(y.T,alpha)/hyper[0]
    dobj = np.zeros(4)
    dobj[0] = 0.5*np.trace(np.dot((alphas-Kinv),dK[0,0]))/hyper[0] - 0.5*n*(ubnd_[0] -lbnd_[0]) + 0.5*(ubnd_[0]-lbnd_[0])*np.matmul(y.T,alpha) /hyper[0]
    dobj[1] = 0.5*np.trace(np.dot((alphas-Kinv),dK[1,0]))/hyper[0]
    dobj[2] = 0.5*np.trace(np.dot((alphas-Kinv),dK[2,0]))/hyper[0]
    dobj[3] = 1/hyper[0]*np.dot((yt).T,alpha) * (ubnd_[3]-lbnd_[3])
    return obj,dobj
def centraldiff(id,h,ubnd_,lbnd_):
    vars1  = np.array([ 1.000, 0.5777, 0.2220, 0.6138 ])
    vars2  = np.array([ 1.000, 0.5777, 0.2220, 0.6138 ])
    vars1[id] += h
    vars2[id] -= h
    ubnd1_ = np.array([2.,2.,2.,2.])
    lbnd1_ = np.array([1.,1.,1.,1.])
    res1,_ = NLML(vars1,ubnd1_,lbnd1_)
    res2,_ = NLML(vars2,ubnd_,lbnd_)
    return (res1-res2)/(2*h)
xe = np.array([0,0.4,0.6,1]).reshape(-1,1)
vars  = np.array([ 1.000, 0.5777, 0.2220, 0.6138 ])
ubnd_ = np.array([2.,2.,2.,2.])
lbnd_ = np.array([1.,1.,1.,1.])

check = centraldiff(2,0.00000001,ubnd_,lbnd_)
print check
ubnd_ = np.array([2.,2.,2.,2.])
lbnd_ = np.array([1.,1.,1.,1.])


print (NLML(vars,ubnd_,lbnd_))
#_,dK = RBF(hyper,xe)
#print dK[2,0]
#print hyper[0]
#NLML(hyper)
