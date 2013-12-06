### 
### 
### LIKELIHOOD FOR CENSORED MODEL 
###
### by: Richards, Hogg, and Long 
### date: 12/3/2013 
### 

import numpy as np
from scipy.special import erf

def nll_fixedB_no_cens(params,mu_b,v_b,sin_c,cos_c,sin_uc,cos_uc,f,e):
    cens = False
    A = params[0]
    B = params[1]
    C = params[2]
    ll = log_likelihood_fixed_w(A,B,C,mu_b,v_b,sin_c,cos_c,sin_uc,cos_uc,f,e,cens)
    return -ll



def log_likelihood_fixed_w(A,B,C,mu_b,v_b,sin_c,cos_c,sin_uc,cos_uc,f,e,cens):
    ## get likelihood of censored observations
    ## if not censored obs, skip this (TODO: find nicer way to implement)
    if sin_c.size > 0 and cens:
        mu_c = pred2mu(sin_c,cos_c,A,B,C)
        muf_c = mag2flux(mu_c)
        llc = log_likelihood_censored(muf_c,mu_b,v_b)
        print llc
    else:
        llc = 0.

    ## get likelihood of uncensored observations
    mu = pred2mu(sin_uc,cos_uc,A,B,C)
    muf = mag2flux(mu)
    ll = log_likelihood_uncensored(muf,f,e,mu_b,v_b,cens)
    print ll

    ## TODO:
    ## penalize likelihood for very small values of 
    ## mu_b or v_b, this will prevent having to specify bounds
    ## for algorithm
    return ll + llc

## given times and freq, creates predictors that will be used in
## likelihood, don't do this inside likelihood at fixed freq, b/c
## then have to compute lots of sines and cosines for every call
def times2pred(t,w):
    return np.sin(2*np.pi*t*w), np.cos(2*np.pi*t*w)

## takes output of times2pred and makes a sinusoidal fit to 
## magnitudes
def pred2mu(sins,coss,A,B,C):
    return A*sins + B*coss + C
    
def log_likelihood_censored(mu,mu_b,v_b):
    return np.sum(np.log(1 - gaussian_cdf(mu,mu_b,v_b)))

def log_likelihood_uncensored(mu,f,e,mu_b,v_b,cens):
    ## log-likelihood not censored
    llnc = 0
    if cens:
        llnc = np.sum(np.log(gaussian_cdf(mu,mu_b,v_b)))
    ## log-likelihood of getting particular flux
    llf = np.sum(log_gaussian_pdf(mu,f,e))
    return llf + llnc


oneoversqrt2 = 1./np.sqrt(2)
## TODO develop unit tests for log_gaussian_pdf
## note: this function is proportional to log gaussian pdf, not equal to
def log_gaussian_pdf(x, mean, var):
    if(np.any(var < 1.e-20)):
        print "log_gaussian_pdf is numerically unstable"
        return np.log(1.e-10)
    return -0.5 * (x - mean)**2 / var

def gaussian_cdf(x, mean, var):
    if(np.any(var < 1.e-20)):
        print "log_gaussian_cdf is numerically unstable"
        return np.log(1.e-10)
    return .5*(1. + erf(oneoversqrt2 * (x - mean)/np.sqrt(var)) ) 


def mag2flux(m, f0 = 1.e6, lim = -10.):
    if(np.min(m) < lim):
        #print "mag below brightness limit; altering"
        m[np.where(m < lim)[0]] = lim
    return f0 * 10**(-0.4*m)

def flux2mag(f, f0 = 1.e6):
    return -2.5 * np.log10(f/f0)


## TODO: figure out why this formula
def magerr2fluxerr(m, merr, f0 = 1.e6):
    #return 0.5 * (mag2flux(m - merr, f0) - mag2flux(m + merr, f0))
    return f0 * 0.4 * np.log(10) * 10**(-0.4 * m) * merr


if __name__ == "__main__":
    assert(np.abs(1. - gaussian_cdf(20,0.332,1.323)) < 1.e-7)
    assert(np.abs(gaussian_cdf(-20,0.332,1.323)) < 1.e-7)
    assert(np.abs(gaussian_cdf(10.1,10.1,4) - .5) < 1.e-7)
    print 'Gaussian cdf tests passed'
    assert(np.abs(flux2mag(mag2flux(10.123)) - 10.123) < 1.e-7)
    assert(np.abs(flux2mag(mag2flux(14.3)) - 14.3) < 1.e-7)
    assert(magerr2fluxerr(10.1, 0.05) > 0)
    print 'Unit tests flux-mag conversions passed'

