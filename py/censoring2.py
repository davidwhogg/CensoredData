"""
This file is part of the Censored Data Project
Copyright 2012, Joseph W. Richards, David W. Hogg, James Long

This code is synchronized with the document tex/ms.tex

Issues:
- limits on gamma integrations are made up (sensitivity not checked)
- make it possible to insert estimated (or known) upper limits
"""

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

    def __init__(self, t, m, em, tc, tolquad = 0.1):
        """ 
        t: times when not censored (1-d ndarray)
        f: flux when not censored (1-d ndarray)
        e: reported error when not censored (1-d ndarray)
        tc: times of censored observations (1-d ndarray)
        """      
        self.tolquad = tolquad # numerical tolerance in integration
        # insert data 
        self.t = t
        self.m = m
        self.em = em
        self.tc = tc

        self.f = mag2flux(self.m)
        self.ef = magerr2fluxerr(self.m, self.em)
        self.ef2 = self.ef**2

        unittests()
        
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
            return  gaussian_cdf(B, ui, VB + sig2 + su2) * gamma_pdf(sig2,S,VS)
        return np.log([integrate.quad(sig_integrand,0.,S+5*np.sqrt(VS),(ui,su2,B,VB,S,VS),epsabs=self.tolquad)[0] \
                       for ui in self.uc])

    def loglikelihood_observed(self,su2,B,VB,Vsig,S,VS):
        def integrand(sig2,ui,fi,ei2,su2,B,VB,Vsig,S,VS):
            p_not_cens = gaussian_cdf(fi, B, VB)
            p_flux = gaussian_pdf(fi, ui, sig2 + su2)
            p_si = gamma_pdf(ei2,sig2,Vsig)
            p_sig2 = gamma_pdf(sig2,S,VS)
            return p_not_cens * p_flux * p_si * p_sig2
        return np.log([integrate.quad(integrand,0.,S+5.*np.sqrt(VS),(ui,fi,ei2,su2,B,VB,Vsig,S,VS),epsabs=self.tolquad)[0]\
                       for (ui,fi,ei2) in zip(self.u,self.f,self.ef2)])

    def negll(self,par):
        return -1*self.log_likelihood(par)

    def optim_fmin(self,p0,maxiter=1000,ftol=0.0001,xtol=0.0001):
        opt = op.fmin(self.negll, p0, maxiter=maxiter,ftol=ftol)
        #opt = op.fmin_bfgs(self.negll, p0, gtol=ftol, maxiter=maxiter)
        return opt

# PDFs and CDFs of normal and Gamma distributions
oneoversqrt2pi = 1./np.sqrt(2.*np.pi)
oneoversqrt2 = 1./np.sqrt(2)

def gaussian_pdf(x, mean, var):
    return (oneoversqrt2pi/np.sqrt(var) * np.exp(-0.5 * (x - mean)**2 / var) )    

def gaussian_cdf(x, mean, var):
    return .5*(1. + erf(oneoversqrt2 * (x - mean)/np.sqrt(var)) ) # look up correct form

def gamma_pdf(x,mean,var):
    theta = var / mean
    k = mean / theta
    return (1 / ((theta**k)*gamma(k))) *  (x**(k-1.)) * np.exp(-x / theta) # look this up (normalization)

def gamma_cdf(x,mean,var):
    theta = var / mean
    k = mean / theta
    return  gammainc(k,x/theta) ## wikipedia and scipy define imcomplete gamma differently

# flux to mag conversions

def mag2flux(m, f0 = 1.e-6):
    return f0 * 10**(-0.4*m)

def flux2mag(f, f0 = 1.e-6):
    return -2.5 * np.log10(f/f0)

def magerr2fluxerr(m, merr, f0 = 1.e-6):
    return 0.5 * (mag2flux(m - merr, f0) - mag2flux(m + merr, f0))

def unittests():
    assert(np.abs(1. - integrate.quad(gaussian_pdf,-20.,20.,(1.322,1.532))[0]) < 1.e-7)
    assert(np.abs(1. - integrate.quad(gamma_pdf,0,50.,(1.322,1.532))[0]) < 1.e-7)
    print 'Unit tests PDFs: All tests passed'
    assert(np.abs(1. - gaussian_cdf(20,0.332,1.323)) < 1.e-7)
    assert(np.abs(gaussian_cdf(-20,0.332,1.323)) < 1.e-7)
    assert(np.abs(1. - gamma_cdf(100,0.332,1.323)) < 1.e-7)
    assert(np.abs(gamma_cdf(0,0.332,1.323)) < 1.e-7)
    print 'Unit tests CDFs: All tests passed'
    assert(np.abs(flux2mag(mag2flux(10.123)) - 10.123) < 1.e-7)
    assert(np.abs(flux2mag(mag2flux(14.3)) - 14.3) < 1.e-7)
    assert(magerr2fluxerr(10.1, 0.05) > 0)
    print 'Unit tests flux-mag conversions passed'
    return None




