###
### implements censoring model
### date April 10, 2012
###

import numpy as np
import scipy.stats as stats
#from scipy.stats import norm as scipynorm
#from scipy.stats import gamma as scipygamma
from scipy import integrate



def newgamma(mean,var):
    return stats.gamma(mean**2/var, scale = var/mean)

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
    return np.sum(loglikelihood_censored(uc,su2,B,VB,S,VS)) + \
         np.sum(loglikelihood_observed(u,f,e,su2,B,VB,Vsig,S,VS))
    
def loglikelihood_censored(uc,su2,B,VB,S,VS):
    ## integrand of equation (7), integrate this across bi
    def erf_integral(bi,sig2,uc,su2,B,VB):
        return stats.norm.cdf(uc, bi, np.sqrt(sig2 + su2)) \
               * stats.gamma.pdf(bi,B**2/VB, scale = VB/B)
    ## integrand of equation (12), integrate this across sig2
    def sig_integrand(sig2,uc,su2,B,VB,S,VS):
        return integrate.quad(erf_integral,0.,B+5.*np.sqrt(VB), (sig2,uc,su2,B,VB))[0] \
               * stats.gamma.pdf(sig2,S**2/VS, scale = VS/S)
    return np.log([integrate.quad(sig_integrand,0.,S+5*np.sqrt(VS),(ui,su2,B,VB,S,VS))[0] for ui in uc])

def loglikelihood_observed(u,f,e,su2,B,VB,Vsig,S,VS):
    def integrand(sig2,u,f,e,su2,B,VB,Vsig,S,VS):
        p_not_cens = stats.gamma.cdf(f,B**2/VB, scale = VB/B)
        p_flux = stats.norm.pdf((f - u) / np.sqrt(sig2 + su2))
        p_si = stats.gamma.pdf(e, sig2**2/Vsig, scale = Vsig/sig2)
        p_sig2 = stats.gamma.pdf(sig2,S**2/VS, scale = VS/S)
        return p_not_cens * p_flux * p_si * p_sig2
    return np.log([integrate.quad(integrand,0.,S+5.*np.sqrt(VS),(ui,fi,ei,su2,B,VB,Vsig,S,VS))[0] for (ui,fi,ei) in zip(u,f,e)])

### TODO: 1. intelligently choose limits for gamma integral
###       2. check that my gamma interpretation is right
###       3. audit all gamma and norm calls
