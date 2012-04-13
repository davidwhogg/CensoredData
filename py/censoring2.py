"""
This file is part of the Censored Data Project
Copyright 2012, Joseph W. Richards, David W. Hogg, James Long

This code is synchronized with the document tex/ms.tex

Issues:
- limits on gamma integrations are made up (sensitivity not checked)
- make it possible to insert estimated (or known) upper limits
- need a folded version of the plot
- need a magnitude version of the plot
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

    def __init__(self, t, m, em, tc, name = "Awesome LC", tolquad = 0.1):
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
        self.name = name
        
        unittests()
        return None

    def mu(self, times, omega, A0, A1, B1):
        return A0 + A1 * np.cos(omega * times) + B1 * np.sin(omega * times)
        
    def log_likelihood(self,par):
        """ 
        computes log p(D|theta,I)
        everything else: same notation as paper (all floats)
        hella ugly unpacking from par[]; you're so hella NorCal
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
        self.u  = self.mu(self.t,  w, A0, A1, B1)
        self.uc = self.mu(self.tc, w, A0, A1, B1)

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

    def get_init_par(self,Period):
        # params:     P,A0,A1,B1,su2,B,VB,Vsig,S,VS
        par = np.zeros(10)
        par[0] = Period
        par[1:4] = self.least_sq(Period)
        par[4] = (0.2 * par[1])**2
        par[5] = 2.*np.abs(np.min(self.f))
        par[6] = par[5]**2
        par[7] = np.median(self.ef2)
        par[8] = par[7]
        par[9] = par[8]**2.
        return par

    def least_sq(self,Period):
        Amat = np.zeros( (3,len(self.t)))
        Amat[0,:] = 1.
        Amat[1,:] = self.mu(self.t,2.*np.pi / Period, 0,1,0)
        Amat[2,:] = self.mu(self.t,2.*np.pi / Period, 0,0,1)
        Atb = np.dot(Amat, self.f)
        print Atb.shape
        AtAinv = np.matrix(np.dot(Amat,Amat.T)).I
        return np.dot(AtAinv,Atb)
        
    def optim_fmin(self,p0,maxiter=1000,ftol=0.0001,xtol=0.0001):
        opt = op.fmin(self.negll, p0, maxiter=maxiter,ftol=ftol)
        #opt = op.fmin_bfgs(self.negll, p0, gtol=ftol, maxiter=maxiter)
        return opt

    def plot(self, ax, par, fold = False):
        '''
        input:
        - ax: matplotlib axes object
        - par: set of hyperparameters

        usage:
            plt.clf()
            ax = plt.subplot(111)
            cens.plot(ax, par)
            plt.savefig('wtf.png')
        '''
        def hogg_errorbar(ax, x, y, yerr, color='k', alpha=0.25):
            for xi,yi,yerri in zip(x,y,yerr):
                ax.plot([xi, xi], [yi - yerri, yi + yerri], color+'-', alpha=alpha)
            return None
        def phase(t, P):
            return (t/P) % 1.
        if(fold):
            x, xc = phase(self.t, par[0]), phase(self.tc, par[0])
            mediant = 0
            tlim = np.array([0., 2.])
            tp = np.linspace(0., par[0]*2., 10000)
            xp = np.linspace(0., 2., len(tp))
        else:
            x, xc = self.t, self.tc
            alltimes = np.append(self.t, self.tc)
            mediant = np.round(np.median(alltimes)).astype(int)
            tlim = np.array([np.min(alltimes), np.max(alltimes)])
            tp = np.linspace(tlim[0], tlim[1], 10000)
            xp = tp
        omega = 2. * np.pi / par[0]
        A0 = par[1]
        A1 = par[2]
        B1 = par[3]
        s2mu = par[4]
        ax.axhline(0., color='k', alpha=0.25)
        ax.plot(x - mediant, self.f, 'ko', alpha=0.5, mec='k')
        hogg_errorbar(ax, x - mediant, self.f, self.ef)
        ax.plot(xc - mediant, np.zeros_like(xc), 'r.', alpha=0.5, mec='r')
        if(fold):
            ax.plot(x - mediant + 1., self.f, 'ko', alpha=0.5, mec='k')
            hogg_errorbar(ax, x - mediant + 1., self.f, self.ef)
            ax.plot(xc - mediant + 1., np.zeros_like(xc), 'r.', alpha=0.5, mec='r')
        
        mup = self.mu(tp, omega, A0, A1, B1)
        ax.plot(xp - mediant, mup + np.sqrt(s2mu), 'b-', alpha=0.25)
        ax.plot(xp - mediant, mup,                 'b-', alpha=0.50)
        ax.plot(xp - mediant, mup - np.sqrt(s2mu), 'b-', alpha=0.25)
        ax.set_xlim(tlim - mediant)
        foo = np.max(self.f + self.ef)
        ax.set_ylim(-0.1 * foo, 1.1 * foo)
        ax.set_xlabel(r'time $t$ (MJD - %d~d)' % mediant)
        ax.set_ylabel(r'flux $f$ ($\mu$Mgy)')
        ax.set_title(self.name)
        return None

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

def mag2flux(m, f0 = 1.e6):
    return f0 * 10**(-0.4*m)

def flux2mag(f, f0 = 1.e6):
    return -2.5 * np.log10(f/f0)

def magerr2fluxerr(m, merr, f0 = 1.e6):
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
