###
### implements censoring model
### date April 10, 2012
###

import numpy as np
from scipy.stats import norm
from scipy.stats import gamma

def log_likelihood(t,f,e,tc,w,A0,A1,B1,su2,B,VB,Vsig,S,VS):
    """ 
    computes log p(D|theta,I)
    t: times when not censored (1-d ndarray)
    f: flux when not censored (1-d ndarray)
    e: reported error when not censored (1-d ndarray)
    tc: times of censored observations (1-d ndarray)
    everything else: same notation as paper (all floats)
    """
    ## convert all times to expected flux (i.e. t_i -> u_i)
    u = A0 + A1*np.sin(t*w*2*np.pi) + B1*np.cos(t*w*2*np.pi)
    uc = A0 + A1*np.sin(tc*w*2*np.pi) + B1*np.cos(tc*w*2*np.pi)

    ## compute loglikelihood
    ll = 0
    for i in uc:
        ll += likelihood_censored(i,su2,B,VB)
    for i in np.range(u.size):
        ll += likelihood_observed(u[i],f[i],e[i],su2,B,VB,Vsig,S,VS)
    return ll

def likelihood_censored(uc,su2,B,VB):
    ## integrand of equation (7), integrate this across bi
    def erf_integral(bi,sig2,uc,su2,B,VB):
        return norm.cdf((bi - uc) / np.sqrt(sig2 + su2)) * (1 / (np.sqrt(2*np.pi*VB))) * np.exp(-(bi-B)**2/(2*VB))
    ## integrand of equation (12), integrate this across sig2
    def sig_integral(sig2,uc,su2,B,VB):
        return integrate.quad(erf_integral,B-3*np.sqrt(VB),B+3*np.sqrt(VB), [sig2,uc,su2,B,VB]) * (1/VB)*gamma.pdf(sig2,B/VB)
    return integrate.quad(sig_integral,0.0,10,[uc,su2,B,VB])
    
def liklihood_observed(u,f,e,su2,B,VB,Vsig,S,VS):
    def integrand(sig2,u,f,e,su2,B,VB,Vsig,S,VS):
        p_not_cens = norm.cdf((f - B) / np.sqrt(VB))
        p_flux = norm.pdf((f - u) / np.sqrt(sig2 + su2))
        p_si = (1/Vsig)*gamma.pdf(e,sig2/Vsig)
        p_sig2 = (1/VB)*gamma.pdf(sig2,B/VB)
        return p_not_cens * p_flux * p_si * p_sig2
    return integrate.quad(integrand,0.0,10,[u,f,e,su2,B,VB,Vsig,S,VS])


### TODO: 1. intelligently choose limits for gamma integral
###       2. check that my gamma interpretation is right

