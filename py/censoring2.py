###
### implements censoring model
### date April 10, 2012

### Updated by JWR on 20120411
###

import numpy as np
#import scipy.stats as stats
#from scipy.stats import norm as scipynorm
#from scipy.stats import gamma as scipygamma
#from scipy.stats import norm
from scipy import integrate
from scipy.special import gamma
from scipy.special import erf
from scipy.special import gammainc
import scipy.optimize as op

class Censored:

    def __init__(self, t, f, e, tc, tolquad = 0.1):
        """ 
        t: times when not censored (1-d ndarray)
        f: flux when not censored (1-d ndarray)
        e: reported error when not censored (1-d ndarray)
        tc: times of censored observations (1-d ndarray)
        """      
        self.tolquad = tolquad # numerical tolerance in integration
        # insert data 
        self.t = t
        self.f = f
        self.e = e
        self.tc = tc
        
    def log_likelihood(self,par):
        """ 
        computes log p(D|theta,I)
        everything else: same notation as paper (all floats)
        """
        P = par[0]
        A0 = par[1]
        A1 = par[2]
        B1 = par[3]
        su2 = par[4]
        B = par[5]
        VB = par[6]
        Vsig = par[7]
        S = par[8]
        VS = par[9]
        
    ## convert all times to expected flux (i.e. t_i -> u_i)
        w =  (2 * np.pi)/ P 
        self.u = A0 + A1*np.cos(self.t*w) + B1*np.sin(self.t*w)
        self.uc = A0 + A1*np.cos(self.tc*w) + B1*np.sin(self.tc*w)

    ## compute loglikelihood
        return np.sum(self.loglikelihood_censored(su2,B,VB,S,VS)) + \
               np.sum(self.loglikelihood_observed(su2,B,VB,Vsig,S,VS))
    
    def loglikelihood_censored(self,su2,B,VB,S,VS):
    ## integrand of equation (12), integrate this across sig2
        def sig_integrand(sig2,ui,su2,B,VB,S,VS):
            return  gaussian_cdf((B-ui) / np.sqrt(VB + sig2 + su2)) * \
                gamma_pdf(sig2,S,VS)
        return np.log([integrate.quad(sig_integrand,0.,S+5*np.sqrt(VS),(ui,su2,B,VB,S,VS),epsabs=self.tolquad)[0] \
                       for ui in self.uc])

    def loglikelihood_observed(self,su2,B,VB,Vsig,S,VS):
        def integrand(sig2,ui,fi,ei,su2,B,VB,Vsig,S,VS):
            p_not_cens = gamma_cdf(fi,B,VB)#fi,B**2/VB, scale = VB/B)
            p_flux = gaussian_pdf((fi - ui) / np.sqrt(sig2 + su2))
            p_si = gamma_pdf(ei,sig2,Vsig)
            p_sig2 = gamma_pdf(sig2,S,VS)
            return p_not_cens * p_flux * p_si * p_sig2
        return np.log([integrate.quad(integrand,0.,S+5.*np.sqrt(VS),(ui,fi,ei,su2,B,VB,Vsig,S,VS),epsabs=self.tolquad)[0]\
                       for (ui,fi,ei) in zip(self.u,self.f,self.e)])

    def negll(self,par):
        return -1*self.log_likelihood(par)

    def optim_fmin(self,p0,maxiter=1000,ftol=0.0001,xtol=0.0001):
        opt = op.fmin(self.negll, p0, maxiter=maxiter,ftol=ftol)
        #opt = op.fmin_bfgs(self.negll, p0, gtol=ftol, maxiter=maxiter)
        return opt


def gaussian_pdf(x):
    return (1 / (np.sqrt(2.*np.pi))) * np.exp(-(x**2) / (2.))    

def gaussian_cdf(x):
    return .5*(1. + erf(x/np.sqrt(2.)))

def gamma_pdf(x,mean,var):
    theta = var / mean
    k = mean / theta
    return (1 / ((theta**k)*gamma(k))) *  (x**(k-1.)) * np.exp(-x / theta)

def gamma_cdf(x,mean,var):
    theta = var / mean
    k = mean / theta
    return  gammainc(k,x/theta) ## wikipedia and scipy define imcomplete gamma differently



### TODO: 1. intelligently choose limits for gamma integral
###       2. check that my gamma interpretation is right; DONE
###       3. audit all gamma and norm calls; DONE, replaced by hand-built functions
