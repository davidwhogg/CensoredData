"""
This file is part of the Censored Data Project
Copyright 2012, Joseph W. Richards, David W. Hogg, James Long

This code is synchronized with the document tex/ms.tex

Issues:
- limits on gamma integrations are made up (sensitivity not checked)
- make it possible to insert estimated (or known) upper limits
- need a magnitude version of the plot
- quad returns nan when VB and VS get very large; consider rescaling fluxes to order unity
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
        
    def log_likelihood(self, par, fast = True):
        """ 
        computes log p(D|theta,I)
        everything else: same notation as paper (all floats)
        hella ugly unpacking from par[]
        """
        P = par[0]
        A0 = par[1]
        A1 = par[2]
        B1 = par[3]
        eta2 = par[4]**2
        B = par[5]
        VB = par[6]**2
        Vsig = par[7]**2
        S = par[8]
        VS = par[9]**2

    ## convert all times to expected flux (i.e. t_i -> u_i)
        w =  (2 * np.pi)/ P 
        self.u  = mag2flux(self.mu(self.t,  w, A0, A1, B1))
        self.uc = mag2flux(self.mu(self.tc, w, A0, A1, B1))

        if(fast):
            return np.sum(self.loglikelihood_censored(eta2,B,VB,S,VS)) + \
                   np.sum(self.loglikelihood_observed_fast(eta2,B,VB,Vsig))
        return np.sum(self.loglikelihood_censored(eta2,B,VB,S,VS)) + \
               np.sum(self.loglikelihood_observed(eta2,B,VB,Vsig,S,VS))
    
    def loglikelihood_censored(self,eta2,B,VB,S,VS):
        """
        likelihood function for censored data
        integrand of equation (12), integrated across sig2
        """
        def sig_integrand(sig2,ui,eta2,B,VB,S,VS):
            #print 'gauss: ' +  str(gaussian_cdf(B, ui, VB + sig2 + eta2 * ui * ui))
            #print gamma_pdf(sig2,S,VS)
            return gaussian_cdf(B, ui, VB + sig2 + eta2 * ui * ui) * gamma_pdf(sig2,S,VS)
        return np.log([integrate.quad(sig_integrand,np.max([0.,S-3.*np.sqrt(VS)]),S+3.*np.sqrt(VS),(ui,eta2,B,VB,S,VS),\
                                      epsabs=self.tolquad)[0] for ui in self.uc])

    
    def loglikelihood_observed_fast(self,eta2,B,VB,Vsig):
        """
        Likelihood function for observed data in the case we believe
        that the reported errors are correct 
        """
        p_not_cens = gaussian_cdf(self.f, B, VB)
        p_flux = gaussian_pdf(self.f, self.u, self.ef2 + eta2 * self.u**2)
        return np.log(p_not_cens * p_flux)

    def loglikelihood_observed(self,eta2,B,VB,Vsig,S,VS):
        """
        log likelihood function for observed data, marginalizing over the sig2
        loglikelihood_observed_fast was created because this function fails for large values of VS
        """
        def integrand(sig2,ui,fi,ei2,eta2,B,VB,Vsig,S,VS):
            p_not_cens = gaussian_cdf(fi, B, VB)
            p_flux = gaussian_pdf(fi, ui, sig2 + eta2 * ui * ui)
            p_si = gamma_pdf(ei2,sig2,Vsig)
            p_sig2 = gamma_pdf(sig2,S,VS)
            return p_not_cens * p_flux * p_si * p_sig2
        return np.log([integrate.quad(integrand,np.max([0.,S-3.*np.sqrt(VS)]),S+3.*np.sqrt(VS),(ui,fi,ei2,eta2,B,VB,Vsig,S,VS),\
                                      epsabs=self.tolquad)[0] for (ui,fi,ei2) in zip(self.u,self.f,self.ef2)])


    def negll(self, par, fast=True):
        """
        negative log-likelihood function for use with fmin
        """
        return -1*self.log_likelihood(par, fast=fast)

    def get_init_par(self,Period):
        """
        quick guess for initial parameters employing least squares
        """
        # params:     P,A0,A1,B1,eta2,B,VB,Vsig,S,VS
        par = np.zeros(10)
        par[0] = Period
        par[1:4] = self.least_sq(Period)
        par[4] = 0.2
        par[5] = 2.*np.abs(np.min(self.f))
        par[6] = par[5]
        par[7] = np.median(self.ef2)
        par[8] = par[7]
        par[9] = par[8] # must be kept small or integration will take too long
        # also if par[9] is too large, the integration may return nan
        return par

    def least_sq(self,Period):
        """
        least squares fit for A0, A1, B1 at a fixed (given) period
        """
        Amat = np.zeros( (3,len(self.t)))
        Amat[0,:] = 1.
        Amat[1,:] = self.mu(self.t,2.*np.pi / Period, 0,1,0)
        Amat[2,:] = self.mu(self.t,2.*np.pi / Period, 0,0,1)
        Atb = np.dot(Amat, self.m) # JWR changed to fitting in mag space
        AtAinv = np.matrix(np.dot(Amat,Amat.T)).I
        return np.dot(AtAinv,Atb)
        
    def optim_fmin(self, p0, maxiter=1000, ftol=0.0001, xtol=0.0001, mfev=1.e8, fast=True):
        """
        maximize the log-likelihood function with respect to the 9 model parameters
        """
        opt = op.fmin(self.negll, p0, args = (fast,), maxiter=maxiter,ftol=ftol,maxfun=mfev)
        #opt = op.fmin_bfgs(self.negll, p0, gtol=ftol, maxiter=maxiter)
        return opt

    def plot(self, ax, par, fold = False, plot_model = True, mag = False):
        '''
        input:
        - ax: matplotlib axes object
        - par: set of hyperparameters
        - fold: if True plot folded light curve, else unfolded
        - plot_model: if True plot light curve model
        - mag: if True plot in mag space, else in flux space

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
            mediant = 0.
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
        eta2 = par[4]**2
        ax.axhline(0., color='k', alpha=0.25)
        if(mag):
            y = self.m
            ey = self.em
            yc = np.zeros_like(xc) + np.max(y) + 0.5
        else:
            y = self.f
            ey = self.ef
            yc = np.zeros_like(xc)
        ax.plot(x - mediant, y, 'ko', alpha=0.5, mec='k')
        hogg_errorbar(ax, x - mediant, y, ey)
        ax.plot(xc - mediant, yc, 'r.', alpha=0.5, mec='r')
        if(fold): # if fold, plot 2 phases of LC
            ax.plot(x - mediant + 1., y, 'ko', alpha=0.5, mec='k')
            hogg_errorbar(ax, x - mediant + 1., y, ey)
            ax.plot(xc - mediant + 1., yc, 'r.', alpha=0.5, mec='r')
        if(plot_model):
            if(mag):
                mup = self.mu(tp, omega, A0, A1, B1)
                mup_p = mup + 1.086 * np.sqrt(eta2)
                mup_m = mup - 1.086 * np.sqrt(eta2)
            else:
                mup = mag2flux(self.mu(tp, omega, A0, A1, B1))
                mup_p = mup * (1. + np.sqrt(eta2))
                mup_m = mup * (1. - np.sqrt(eta2))
            ax.plot(xp - mediant, mup_p, 'b-', alpha=0.25)
            ax.plot(xp - mediant, mup, 'b-', alpha=0.50)
            ax.plot(xp - mediant, mup_m, 'b-', alpha=0.25)
        ax.set_xlim(tlim - mediant)
        if(mag):
            ax.set_ylim(np.max(y) + 1., np.min(y) - 0.5)
            ax.set_ylabel(r'magnitude $m$ (mag)')
        else:
            foo = np.max(y + ey)
            ax.set_ylim(-0.1 * foo, 1.1 * foo)
            ax.set_ylabel(r'flux $f$ ($\mu$Mgy)')
        if(fold):
            ax.set_xlabel(r'$\phi$')
        else:
            ax.set_xlabel(r'time $t$ (MJD - %d~d)' % mediant)
        ax.set_title(self.name)
        return None


##################################
# PDFs and CDFs of normal and Gamma distributions
oneoversqrt2pi = 1./np.sqrt(2.*np.pi)
oneoversqrt2 = 1./np.sqrt(2)

def gaussian_pdf(x, mean, var):
    #if(np.any(var <= 0)):
    #    print x, mean, var
    #    raise Exception
    return (oneoversqrt2pi/np.sqrt(var) * np.exp(-0.5 * (x - mean)**2 / var) )

def gaussian_cdf(x, mean, var):
    return .5*(1. + erf(oneoversqrt2 * (x - mean)/np.sqrt(var)) ) # look up correct form

def gamma_pdf(x,mean,var):
    theta = var / mean
    k = mean / theta
    pdf = (1. / ((theta**k)*gamma(k))) *  (x**(k-1.)) * np.exp(-x / theta)
    if pdf == np.inf:
        return 1.
    return  pdf # look this up (normalization)

def gamma_cdf(x,mean,var):
    theta = var / mean
    k = mean / theta
    return  gammainc(k,x/theta) ## wikipedia and scipy define imcomplete gamma differently

################################
# flux to mag conversions

def mag2flux(m, f0 = 1.e6):
    return f0 * 10**(-0.4*m)

def flux2mag(f, f0 = 1.e6):
    return -2.5 * np.log10(f/f0)

def magerr2fluxerr(m, merr, f0 = 1.e6):
    #return 0.5 * (mag2flux(m - merr, f0) - mag2flux(m + merr, f0))
    return f0 * 0.4 * np.log(10) * 10**(-0.4 * m) * merr

# unit tests
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
