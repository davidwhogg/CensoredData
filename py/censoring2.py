###
### implements censoring model
### date April 10, 2012
###

import numpy as np
from scipy.stats import norm
from scipy.stats import gamma

def log_likelihood(t,f,e,tc,w,A0,A1,B1,su2,B,VB,Vsig,S,VS):
    """ 
    computes log(p(D|theta,I))
    t: times when not censored (1-d ndarray of length n)
    f: flux when not censored (1-d ndarray of length n)
    e: reported error when not censored (1-d ndarray of length n)
    tc: times of censored observations (1-d ndarray of length m)
    everything else: exactly as in paper (all floats)
    """
    ## convert all times to expected flux (i.e. t_i -> u_i)
    u = A0 + A1*np.sin(t*w*2*np.pi) + B1*np.cos(t*w*2*np.pi)
    uc = A0 + A1*np.sin(tc*w*2*np.pi) + B1*np.cos(tc*w*2*np.pi)

    ## compute loglikelihood
    ll = 0
    for i in uc:
        ll += likelihood_censored(i,su2,B,VB)
    for i in u:
        ll += likelihood_observed()
    return ll

def likelihood_censored(uc,su2,B,VB):
    ## integrand of equation (7), integrate this across bi
    def erf_integral(bi,sig2,uc,su2,B,VB):
        return norm.cdf((bi - uc) / np.sqrt(sig2 + su2)) * (1 / (np.sqrt(2*np.pi*VB))) * np.exp(-(bi-B)**2/(2*VB))
    ## integrand of equation (12), integrate this across sig2
    def sig_integral(sig2,uc,su2,B,VB):
        return integrate.quad(erf_integral, 0.5, 1.5, [sig2,uc,su2,B,VB]) * (1/VB)*gamma.pdf(sig2,B/VB)
    return integrate.quad(sig_integral,.5,1.5,[uc,su2,B,VB])
    
def liklihood_observed(c):
    print "hello"
