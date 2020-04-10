import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import random
plt.matplotlib.rc('xtick', labelsize=12)
plt.matplotlib.rc('ytick', labelsize=12)
plt.rcParams.update({'font.size': 16})
################################################################################
# General Gaussian Process Regression, Rasmussen
################################################################################
class GPR:
    """
    Gaussian Process Regressor
    X           : nxdimension array of data
    y           : nx1 array of targets of data
    noise_var   : noise variance of the data
    noise_fix   : noise variance is fixed
    stab        : added to cov.mat. for stability
    bound       : bounds for hyperparameter optimization
    params      : array with hyperparameters
    LML         : log marginal likelihood
    """
    # Initialize
    def __init__(self, X, y, noise_var=None, noise_fix=False):
        self.dim        = X.shape[1]
        self.N          = X.shape[0]
        self.noise      = noise_var
        self.noise_fix  = noise_fix
        self.X          = X
        self.y          = y
        self.stab       = 1e-6
        self.bound      = ()
        self.params     = self.hyperparams()
        self.LML        = self.likelihood(self.params)
    # Initialize hyperparamters for optimization
    def hyperparams(self):
        hyper           = np.ones(self.dim+1)
        self.id_theta   = np.arange(hyper.shape[0])
        for i in range(0,self.dim+1):
            self.bound  += ((1e-6,None),)
        if self.noise is not None and self.noise_fix is False:
            sigma_n = np.array([self.noise])
            hyper      = np.concatenate([hyper,sigma_n])
            self.bound += ((1e-6,None),)
        return hyper

    # Create RBF covariance matrix
    def RBF(self,hyper,xi,xj=None):
        if xj is None:
            xj = xi
        sigma_f     = np.array(hyper[0])
        lengthscale = np.array(hyper[1:])
        r           = np.expand_dims(xi*lengthscale,1) - \
                      np.expand_dims(xj*lengthscale,0)
        return sigma_f * np.exp(-0.5 * np.sum(r**2,axis=2))

    # Objective function to be minimized
    def likelihood(self,hyper):
        if self.noise is not None and self.noise_fix is False:
            sigma_n  = hyper[-1]
        else:
            sigma_n = 0.
        theta   = hyper[self.id_theta];self.theta = theta

        K = self.RBF(theta,self.X)+np.eye(self.N)*sigma_n
        L = np.linalg.cholesky(K+np.eye(self.N)*self.stab);self.L = L

        alpha = np.linalg.solve(L.T,np.linalg.solve(L,self.y))
        LML   = -0.5 * np.matmul(self.y.T,alpha)  - np.sum(np.log(np.diag(L))) - \
                0.5 * np.log(2.*np.pi) * self.N
        self.LML = LML
        return -LML

    # Optimize the hyperparamters
    def optimize(self,restart=None):
        if restart is None:
            res = minimize(value_and_grad(self.likelihood), self.params, bounds=self.bound,
                            jac=True, method='L-BFGS-B',callback=self.likelihood)
            self.params = res.x
        else:
            counter = 0
            obj     = self.LML
            while (counter <= restart):
                self.params = np.random.rand(self.params.size) * random.randint(0,3)
                res = minimize(value_and_grad(self.likelihood), self.params, bounds=self.bound,
                                jac=True, method='L-BFGS-B',callback=self.likelihood)
                self.params = res.x
                counter += 1
                if res.fun < -self.LML:
                    obj = res.fun
                    self.params = res.x

    # Making predictions
    def inference(self,x,return_std=False):
        k_s   = self.RBF(self.theta,x,self.X)
        k_ss  = self.RBF(self.theta,x,x)
        alpha = np.linalg.solve(self.L.T,np.linalg.solve(self.L,self.y))
        mean  = np.matmul(k_s,alpha)
        v     = np.linalg.solve(self.L,k_s.T)
        var   = k_ss - np.dot(v.T,v)
        std   = np.sqrt(np.diag(var))
        if return_std is False:
            return mean,var
        else:
            return mean,std

    # Plotting tool for predictions
    def plot(self,name,plot_std=False):
        if self.X.shape[1] > 1:
            raise Exception('Dimension of X should be 1 for this method...')
        x = np.linspace(np.min(self.X)-2.,np.max(self.X)+2,100).reshape(-1,1)
        self.optimize(restart=10)
        self.likelihood(self.params)
        mean,std = self.inference(x,return_std=True)
        plt.plot(x,mean,"--",label='GPR-'+str(name), color='deepskyblue')
        if plot_std is True:
            plt.fill_between(x.ravel(),mean.ravel() + 2 * std,mean.ravel() - 2 * std,
                                        alpha=0.2,color='deepskyblue');
            plt.fill_between(x.ravel(),mean.ravel() + 1 * std,mean.ravel() - 1 * std,
                                        alpha=0.3,color='deepskyblue');
            plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.legend()
################################################################################
# Two Fidelity Gaussian Process Regression, Forrester et al./2007
################################################################################
class coGPR():
    """
    Gaussian Process Regressor with two fidelities
    Xc               : nxdimension array of cheap data
    yc               : nx1 array of targets of cheap data
    Xe               : nxdimension array of expensive data
    ye               : nx1 array of targets of expensive data
    noise_var_c     : noise variance of the data cheap data
    noise_fix_c     : noise variance is fixed for cheap data
    noise_var_e     : noise variance of the data cheap data
    noise_fix_e     : noise variance is fixed for cheap data
    stab            : added to cov.mat. for stability
    bound           : bounds for hyperparameter optimization
    params          : array with hyperparameters
    LML             : log marginal likelihood

    """
    def __init__(self, Xc, Xe, yc, ye, noise_var_c=None, noise_fix_c=False,
                    noise_var_e=None, noise_fix_e=False):
        self.dim          = Xc.shape[1]
        self.Nc           = Xc.shape[0]
        self.Ne           = Xe.shape[0]
        self.N            = self.Ne + self.Nc
        self.noise_c      = noise_var_c
        self.noise_fix_c  = noise_fix_c
        self.noise_e      = noise_var_e
        self.noise_fix_e  = noise_fix_e
        self.Xe           = Xe
        self.ye           = ye
        self.Xc           = Xc
        self.yc           = yc
        self.y            = np.vstack((yc,ye))
        self.stab         = 1e-6
        self.bound       = ()
        self.params       = self.hyperparams()
        self.LML          = self.likelihood(self.params)

    # Initialize hyperparameters
    def hyperparams(self):
        # Cheap hyperparameters
        hyper_c = np.ones(self.dim+1)
        self.id_theta_c = np.arange(hyper_c.shape[0])
        for i in range(0,self.dim+1):
            self.bound += ((1e-6,None),)
        if self.noise_c is not None and self.noise_fix_c is False:
            sigma_n_c = np.array([self.noise_c])
            hyper_c      = np.concatenate([hyper_c,sigma_n_c])
            self.bound += ((1e-6,None),)
        # Expensive hyperparameters
        hyper_e = np.ones(self.dim+1)
        for i in range(0,self.dim+1):
            self.bound += ((1e-6,None),)
        self.id_theta_e = np.arange(hyper_c.shape[0],hyper_c.shape[0]+\
                                                               hyper_e.shape[0])
        if self.noise_e is not None and self.noise_fix_e is False:
            sigma_n_e = np.array([self.noise_e])
            hyper_e   = np.concatenate([hyper_e,sigma_n_e])
            self.bound += ((1e-6,None),)
        # Difference hyperparameters
        rho = np.array([1.])
        self.bound      += ((None,None),)
        hyper = np.concatenate([hyper_c,hyper_e,rho])
        return hyper

    # RBF Covariance Matrix
    def RBF(self,hyper,xi,xj=None):
        if xj is None:
            xj = xi
        sigma_f     = hyper[0]
        lengthscale = hyper[1:]
        r           = np.expand_dims(xi*lengthscale,1) - \
                      np.expand_dims(xj*lengthscale,0)
        return sigma_f * np.exp(-0.5 * np.sum(r**2,axis=2))

    # Log Marginal likelihood
    def likelihood(self, hyper):
        if (self.noise_c is not None and self.noise_fix_c is False):
            sigma_n_c = hyper[2]
            if (self.noise_e is not None and self.noise_fix_e is False):
                sigma_n_e = hyper[5]
            else:
                sigma_n_e = 0.
        elif (self.noise_c is None and self.noise_fix_c is False) and \
             (self.noise_e is not None and self.noise_fix_e is False):
                sigma_n_e = hyper[4]
                sigma_n_c    = 0.
        else:
            sigma_n_c    = 1e-6
            sigma_n_e    = 1e-6

        rho = hyper[-1]; self.rho = rho
        theta_c = hyper[self.id_theta_c]; self.theta_c = theta_c
        theta_e = hyper[self.id_theta_e]; self.theta_e = theta_e
        #hyper = 1.
        K_cc = self.RBF(theta_c,self.Xc) + np.eye(self.Nc)*sigma_n_c
        K_ce = rho * self.RBF(theta_c,self.Xc,self.Xe) +\
        np.vstack((np.zeros(((self.Nc-self.Ne),self.Ne)),np.eye(self.Ne)))*\
        sigma_n_c

        K_ee = rho**2 * self.RBF(theta_c,self.Xe) +\
        np.eye(self.Ne)*sigma_n_c + \
        self.RBF(theta_e,self.Xe) + np.eye(self.Ne)*sigma_n_e

        K = np.vstack((np.hstack((K_cc,K_ce)),np.hstack((K_ce.T,K_ee))))

        L = np.linalg.cholesky(K+np.eye(self.N)*self.stab);self.L = L

        alpha = np.linalg.solve(L.T,np.linalg.solve(L,self.y));self.alpha = alpha
        LML   = -0.5 * np.matmul(self.y.T,alpha)  - np.sum(np.log(np.diag(L))) - \
                0.5 * np.log(2.*np.pi) * self.N
        self.LML = LML
        return -LML
    # Optimizing hyperparameters
    def optimize(self,restart=None):
        if restart is None:
            res = minimize(value_and_grad(self.likelihood), self.params, bounds=self.bound,
                            jac=True, method='L-BFGS-B',callback=self.likelihood)
            self.params = res.x
        else:
            counter = 0
            obj     = self.LML
            while (counter <= restart):
                self.params = np.random.rand(self.params.size)
                self.params[-1] = 1
                res = minimize(value_and_grad(self.likelihood), self.params, bounds=self.bound,
                                jac=True, method='L-BFGS-B',callback=self.likelihood)
                self.params = res.x
                counter += 1
                if res.fun < -self.LML:
                    obj = res.fun
                    self.params = res.x

    # Predictions
    def inference(self, x, return_std=False):
        self.likelihood(self.params)
        c_c = self.rho * self.RBF(self.theta_c,self.Xc,x)
        c_e = self.rho**2 * self.RBF(self.theta_c,self.Xe,x) + \
              self.RBF(self.theta_e,self.Xe,x)
        c = np.vstack((c_c,c_e))
        mean = np.matmul(c.T,self.alpha)
        v    = np.linalg.solve(self.L.T,np.linalg.solve(self.L,c))
        var  = self.rho**2 * self.RBF(self.theta_c,x) + \
        self.RBF(self.theta_e,x) - np.matmul(c.T,v)
        std  = np.sqrt(np.diag(var))
        if return_std is False:
            return mean,var
        else:
            return mean,std

    # Plotting tool for predictions
    def plot(self,name,plot_std=False):
        if self.Xe.shape[1] > 1:
            raise Exception('Dimension of Xe and Xc should be 1 for this method...')
        x = np.linspace(np.min(self.Xe),np.max(self.Xe),100).reshape(-1,1)
        self.optimize(restart=9);
        mean,std = self.inference(x,return_std=True)
        plt.plot(x,mean,":",label='coGPR-'+str(name), color='lime')
        if plot_std is True:
            plt.fill_between(x.ravel(),mean.ravel() + 2 * std,mean.ravel() - 2 * std,
                                        alpha=0.2,color='lime')
            plt.fill_between(x.ravel(),mean.ravel() + 1 * std,mean.ravel() - 1 * std,
                                        alpha=0.3,color='lime')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.legend()
################################################################################
# Two Fidelity Gaussian Process Regression, Perdikaris et al./2016
################################################################################
class multiGPR():
    """
    Gaussian Process Regressor with two fidelities
    Xc               : nxdimension array of cheap data
    yc               : nx1 array of targets of cheap data
    Xe               : nxdimension array of expensive data
    ye               : nx1 array of targets of expensive data
    noise_var_c     : noise variance of the data cheap data
    noise_fix_c     : noise variance is fixed for cheap data
    noise_var_e     : noise variance of the data cheap data
    noise_fix_e     : noise variance is fixed for cheap data
    stab            : added to cov.mat. for stability
    bound           : bounds for hyperparameter optimization
    params          : array with hyperparameters
    LML             : log marginal likelihood

    """
    def __init__(self, Xc, Xe, yc, ye, noise_var_c=None, noise_fix_c=False,
                    noise_var_e=None, noise_fix_e=False):
        self.dim          = Xc.shape[1]
        self.Nc           = Xc.shape[0]
        self.Ne           = Xe.shape[0]
        self.N            = self.Ne + self.Nc
        self.noise_c      = noise_var_c
        self.noise_fix_c  = noise_fix_c
        self.noise_e      = noise_var_e
        self.noise_fix_e  = noise_fix_e
        self.Xe           = Xe
        self.ye           = ye
        self.Xc           = Xc
        self.yc           = yc
        self.y            = np.vstack((yc,ye))
        self.stab         = 1e-6
        self.lowreg()
        self.bound        = ()
        self.params       = self.hyperparams()
        self.LML          = self.likelihood(self.params)

    # Initialize hyperparameters
    def hyperparams(self):
        hyper_e = np.ones(self.dim+1)+0.1*np.ones(self.dim+1)
        self.id_theta_e = np.arange(hyper_e.shape[0])
        for i in range(0,self.dim+1):
            self.bound  += ((1e-6,None),)

        if self.noise_e is not None and self.noise_fix_e is False:
            sigma_n_e = np.array([self.noise_e])
            hyper_e   = np.concatenate([hyper_e,logsigma_n_e])
            self.bound  += ((1e-6,None),)

        rho = np.array([1.])
        self.bound  += ((None,None),)
        hyper = np.concatenate([hyper_e,rho])
        return hyper

    # Lower fidelity regression
    def lowreg(self):
        self.model_low = GPR(self.Xc,self.yc,self.noise_c,self.noise_fix_c)
        self.model_low.optimize()
        self.mc,self.covc = self.model_low.inference(self.Xe)

    # RBF Covariance Matrix
    def RBF(self,hyper,xi,xj=None):
        if xj is None:
            xj = xi
        sigma_f     = hyper[0]
        lengthscale = hyper[1:]
        r           = np.expand_dims(xi*lengthscale,1) - \
                      np.expand_dims(xj*lengthscale,0)
        return sigma_f * np.exp(-0.5 * np.sum(r**2,axis=2))

    # log marginal likelihood
    def likelihood(self, hyper):

        if (self.noise_e is not None and self.noise_fix_e is False):
            sigma_n_e = hyper[2]
        else:
            sigma_n_e = 0.

        rho = hyper[-1]; self.rho = rho
        theta_e = hyper[self.id_theta_e]; self.theta_e = theta_e

        K = self.RBF(theta_e,self.Xe) + np.eye(self.Ne) * self.stab

        L = np.linalg.cholesky(K+np.eye(self.Ne)*self.stab);self.L = L

        alpha = np.linalg.solve(L.T,np.linalg.solve(L,(self.ye-rho*self.mc)))

        self.alpha = alpha

        NLML   = 0.5*self.Ne*np.log(hyper[0]) + 0.5*np.sum(np.log(np.diag(L))) +\
                0.5*np.matmul((self.ye-rho*self.mc).T,alpha)/hyper[0]
        return NLML
    # Optimize hyperparameters
    def optimize(self,restart=None):
        if restart is None:
            res = minimize(value_and_grad(self.likelihood), self.params, bounds=self.bound,
                            jac=True, method='L-BFGS-B',callback=self.likelihood)
            self.params = res.x
        else:
            counter = 0
            obj     = self.LML
            while (counter <= restart):
                self.params = np.random.rand(self.params.size)
                res = minimize(value_and_grad(self.likelihood), self.params, bounds=self.bound,
                                jac=True, method='L-BFGS-B',callback=self.likelihood)
                self.params = res.x
                counter += 1
                if res.fun < -self.LML:
                    obj = res.fun
                    self.params = res.x

    # Predictions
    def inference(self, x, return_std=False):
        self.likelihood(self.params)
        m_low,cov_low   = self.model_low.inference(x)
        k_s   = self.RBF(self.theta_e,x,self.Xe)
        k_ss  = self.RBF(self.theta_e,x)
        alpha = np.linalg.solve(self.L.T,np.linalg.solve(self.L,(self.ye-self.rho*self.mc)))
        mean    = self.rho*m_low + np.matmul(k_s,alpha)
        v     = np.linalg.solve(self.L,k_s.T)
        var  = self.rho**2 * cov_low + k_ss - np.dot(v.T,v)
        std  = np.sqrt(np.diag(var))
        if return_std is False:
            return mean,var
        else:
            return mean,std

    # Plotting tool for predictions
    def plot(self,name,plot_std=False):
        if self.Xe.shape[1] > 1:
            raise Exception('Dimension of Xe and Xc should be 1 for this method...')
        x = np.linspace(np.min(self.Xe),np.max(self.Xe),100).reshape(-1,1)
        self.optimize(restart=2);
        mean,std = self.inference(x,return_std=True)
        plt.plot(x,mean,":",label='multiGPR-'+str(name), color='lime')
        if plot_std is True:
            plt.fill_between(x.ravel(),mean.ravel() + 2 * std,mean.ravel() - 2 * std,
                                        alpha=0.2,color='lime')
            plt.fill_between(x.ravel(),mean.ravel() + 1 * std,mean.ravel() - 1 * std,
                                        alpha=0.3,color='lime')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
################################################################################
