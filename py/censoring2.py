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

    def __init__(self, t, m, em, tc, b = None, f0 = 1.e6, name = "Awesome LC", tolquad = 0.1):
        """ 
        t: times when not censored (1-d ndarray)
        f: flux when not censored (1-d ndarray)
        e: reported error when not censored (1-d ndarray)
        tc: times of censored observations (1-d ndarray)
        """      
        self.tolquad = tolquad # numerical tolerance in integration
        self.f0 = f0
        # insert data 
        self.t = t
        self.m = m
        self.em = em
        self.tc = tc
        self.f = mag2flux(self.m, f0=self.f0)
        self.ef = magerr2fluxerr(self.m, self.em, f0=self.f0)
        self.ef2 = self.ef**2
        self.name = name
        self.b = b # known upper limits, if given
        if(self.b != None):
            self.bf = mag2flux(self.b, f0 = self.f0)
        
        unittests()
        return None

    def mu(self, times, omega, A0, A1, B1, A01=0., A02 = 0.):
        return A0 + A01 * times + A02 * times**2 + A1 * np.cos(omega * times) + B1 * np.sin(omega * times)
        
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
        B = np.abs(par[5])
        VB = par[6]**2
        Vsig = par[7]**2
        S = np.abs(par[8])
        VS = par[9]**2
        if len(par) > 10:
            A01 = par[10]
        if len(par) > 11:
            A02 = par[11]

        w =  (2 * np.pi)/ P

        # handle case of no non-detections
        if(len(self.tc) == 0):
            if len(par) == 10:
                self.u  = mag2flux(self.mu(self.t,  w, A0, A1, B1, A01 = 0., A02=0.), f0=self.f0)
            else:
                self.u  = mag2flux(self.mu(self.t,  w, A0, A1, B1, A01, A02=0.), f0=self.f0)
            if(fast):
                return np.sum(self.loglikelihood_observed_fast(eta2,B,VB,Vsig))
            return np.sum(self.loglikelihood_observed(eta2,B,VB,Vsig,S,VS))
 

    ## convert all times to expected flux (i.e. t_i -> u_i)
        if len(par) == 10:
            self.u  = mag2flux(self.mu(self.t,  w, A0, A1, B1, A01 = 0., A02=0.), f0=self.f0)
            self.uc = mag2flux(self.mu(self.tc, w, A0, A1, B1, A01 = 0., A02=0.), f0=self.f0)
        else:
            self.u  = mag2flux(self.mu(self.t,  w, A0, A1, B1, A01, A02=0.), f0=self.f0)
            self.uc = mag2flux(self.mu(self.tc, w, A0, A1, B1, A01, A02=0.), f0=self.f0)

        if(self.b==None): # if no upper limits are given
            if(fast):
                llc = np.sum(self.loglikelihood_censored(eta2,B,VB,S,VS))
                llo = np.sum(self.loglikelihood_observed_fast(eta2,B,VB,Vsig))
                if(np.isnan(llc) or np.isnan(llo)):
                    print 'LL is nan'
                if(np.abs(llc) == np.inf or np.abs(llo) == np.inf):
                    print 'LL is inf or -inf'
                return llc + llo
            return np.sum(self.loglikelihood_censored(eta2,B,VB,S,VS)) + \
                   np.sum(self.loglikelihood_observed(eta2,B,VB,Vsig,S,VS))
        else: # if upper limits are given ONLY FOR CENSORED DATA
            assert(len(self.b) == len(self.tc))
            if(fast):
                return np.sum(self.loglikelihood_bgiven_censored(eta2,S,VS)) + \
                       np.sum(self.loglikelihood_observed_fast(eta2,B,VB,Vsig))
            return np.sum(self.loglikelihood_bgiven_censored(eta2,S,VS)) + \
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
        return np.log([integrate.quad(sig_integrand,np.max([0.,S-5.*np.sqrt(VS)]),S+5.*np.sqrt(VS),(ui,eta2,B,VB,S,VS),\
                                      epsabs=self.tolquad)[0] for ui in self.uc])
   
    def loglikelihood_observed_fast(self,eta2,B,VB,Vsig):
        """
        Likelihood function for observed data in the case we believe
        that the reported errors are correct 
        """
        p_not_cens = gaussian_cdf(self.f, B, VB)
        p_flux = gaussian_pdf(self.f, self.u, self.ef2 + eta2 * self.u**2)
        logarg = p_not_cens * p_flux
        if np.min(logarg) < 1.e-300:
            logarg[np.where(logarg < 1.e-300)[0]] = 1.e-100
        return np.log(logarg)

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
        return np.log([integrate.quad(integrand,np.max([0.,S-5.*np.sqrt(VS)]),S+5.*np.sqrt(VS),(ui,fi,ei2,eta2,B,VB,Vsig,S,VS),\
                                      epsabs=self.tolquad)[0] for (ui,fi,ei2) in zip(self.u,self.f,self.ef2)])

    def loglikelihood_bgiven_censored(self,eta2,S,VS):
        """
        likelihood function for censored data, when bi is given for each censored datum
        integrand of equation (12), integrated across sig2
        """
        def sig_integrand(sig2,bi,ui,eta2,S,VS):
            return gaussian_cdf(bi, ui, sig2 + eta2 * ui * ui) * gamma_pdf(sig2,S,VS)
        #print 'bf: ' + str(self.bf) + ' uc: ' + str(self.uc)
        #print 'S: ' + str(S) + ' VS: ' + str(VS) + ' eta2: ' + str(eta2)
        #print np.log([sig_integrand(S,bi,ui,eta2,S,VS) for (bi,ui) in zip(self.bf,self.uc)])
        return np.log([integrate.quad(sig_integrand,np.max([0.,S-5.*np.sqrt(VS)]),S+5.*np.sqrt(VS),(bi,ui,eta2,S,VS),\
                                      epsabs=self.tolquad)[0] for (bi,ui) in zip(self.bf,self.uc)])
   

    def negll(self, par, fast=True):
        """
        negative log-likelihood function for use with fmin
        """
        return -1*self.log_likelihood(par, fast=fast)

    def get_init_par(self, Period, detrend=0):
        """
        quick guess for initial parameters employing least squares
        """
        # params:     P,A0,A1,B1,eta2,B,VB,Vsig,S,VS
        if(detrend==1):
            par = np.zeros(11)
        elif(detrend==2):
            par = np.zeros(12)
        else:
            par = np.zeros(10)
        par[0] = Period
        ls = self.least_sq(Period, detrend)
        par[1:4] = ls[0:3]
        if(detrend>0):
            par[10] = ls[3]
        if(detrend==2):
            par[11] = ls[4]
        par[4] = 0.2
        par[5] = 2.*np.abs(np.min(self.f))
        par[6] = par[5]
        par[7] = np.median(self.ef2) 
        par[8] = par[7]
        par[9] = par[8] # must be kept small or integration will take too long
        # also if par[9] is too large, the integration may return nan
        return par

    def least_sq(self,Period, detrend=0):
        """
        least squares fit for A0, A1, B1 at a fixed (given) period
        """
        if(detrend==1):
            Amat = np.zeros( (4,len(self.t)))
        elif(detrend==2):
            Amat = np.zeros( (5,len(self.t)))
        else:
            Amat = np.zeros( (3,len(self.t)))
        Amat[0,:] = 1.
        Amat[1,:] = self.mu(self.t,2.*np.pi / Period, 0,1,0)
        Amat[2,:] = self.mu(self.t,2.*np.pi / Period, 0,0,1)
        if(detrend>0):
            Amat[3,:] = self.t
        if(detrend==2):
            Amat[4,:] = self.t**2
        Atb = np.dot(Amat, self.m) 
        AtAinv = np.matrix(np.dot(Amat,Amat.T)).I
        return np.dot(AtAinv,Atb).tolist()[0]
        
    def optim_fmin(self, p0, maxiter=1000, ftol=0.0001, xtol=0.0001, mfev=1.e8, fast=True):
        """
        maximize the log-likelihood function with respect to the 9 model parameters
        fmin is more robust to numerical errors, as it can find its way out of nan's and inf's
        """
        opt = op.fmin(self.negll, p0, args = (fast,), maxiter=maxiter,ftol=ftol,maxfun=mfev)
        return opt

    def optim_fmin_bfgs(self, p0, maxiter=1000, gtol=0.0001, fast=True):
        """
        maximize the log-likelihood function with respect to the 9 model parameters using BFGS
        BFGS often converges in fewer iterations, but will check out on a nan or inf
        """
        opt = op.fmin_bfgs(self.negll, p0, args = (fast,), gtol=gtol, maxiter=maxiter)
        return opt



    def plot(self, ax, par, fold = False, plot_model = True, mag = False, plot_title = True,\
             period = None):
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
        if len(par) > 11:
            A02 = par[11]
        else:
            A02 = 0.
        if len(par) > 10:
            A01 = par[10]
        else:
            A01 = 0.
        ax.axhline(0., color='k', alpha=0.25)
        if(mag):
            y = self.m
            ey = self.em
            if(self.b == None):
                yc = np.zeros_like(xc) + np.max(y) + 0.25
            else:
                yc = self.b
        else:
            y = self.f
            ey = self.ef
            yc = np.zeros_like(xc)
        ax.plot(x - mediant, y, 'ko', alpha=0.5, mec='k', markersize=4)
        hogg_errorbar(ax, x - mediant, y, ey)
        ax.plot(xc - mediant, yc, 'r|', alpha=0.3, mec='r', markersize=10)
        if(fold): # if fold, plot 2 phases of LC
            ax.plot(x - mediant + 1., y, 'ko', alpha=0.5, mec='k', markersize=4)
            hogg_errorbar(ax, x - mediant + 1., y, ey)
            ax.plot(xc - mediant + 1., yc, 'r|', alpha=0.3, mec='r', markersize=10)
        if(plot_model):
            if(mag):
                mup = self.mu(tp, omega, A0, A1, B1, A01, A02)
                mup_p = mup + 1.086 * np.sqrt(eta2)
                mup_m = mup - 1.086 * np.sqrt(eta2)
            else:
                mup = mag2flux(self.mu(tp, omega, A0, A1, B1, A01, A02), f0=self.f0)
                mup_p = mup * (1. + np.sqrt(eta2))
                mup_m = mup * (1. - np.sqrt(eta2))
            ax.plot(xp - mediant, mup, 'b-', alpha=0.80)
            #ax.plot(xp - mediant, mup_p, 'b-', alpha=0.3)
            #ax.plot(xp - mediant, mup_m, 'b-', alpha=0.3)
        ax.set_xlim(tlim - mediant)
        if(mag):
            ax.set_ylim(np.max(y) + 0.5, np.min(y) - 0.5)
            ax.set_ylabel(r'magnitude $m$ (mag)')
        else:
            foo = np.max(y + ey)
            ax.set_ylim(-0.1 * foo, 1.1 * foo)
            ax.set_ylabel(r'flux $f$ ($\mu$Mgy)')
        if(fold):
            ax.set_xlabel(r'$\phi$')
        else:
            ax.set_xlabel(r'time $t$ (MJD $-$ %d d)' % mediant)
        if(plot_title):
            ax.set_title(self.name)
        if(period != None):
            ax.text(0.05, 0.95,'P = %.1f d' % period,
                    horizontalalignment='left',
                    verticalalignment='center',
                    transform = ax.transAxes)
        return None


##################################
# PDFs and CDFs of normal and Gamma distributions
oneoversqrt2pi = 1./np.sqrt(2.*np.pi)
oneoversqrt2 = 1./np.sqrt(2)

def gaussian_pdf(x, mean, var):
    #if(np.any(var <= 0)):
    #    print x, mean, var
    #    raise Exception
    if(np.any(var < 1.e-20)):
        return 1.e-10
    return (oneoversqrt2pi/np.sqrt(var) * np.exp(-0.5 * (x - mean)**2 / var) )

def gaussian_cdf(x, mean, var):
    if(np.any(var < 1.e-20)):
        return 1.e-10
    return .5*(1. + erf(oneoversqrt2 * (x - mean)/np.sqrt(var)) ) 

def gamma_pdf(x,mean,var):
    theta = var / mean
    k = mean / theta
    if(np.log(gamma(k)) == np.inf):
        return 1.e-10
    pdf = np.exp(-k*np.log(theta) - np.log(gamma(k)) + (k-1.)*np.log(x) - x/theta)
    if(pdf == np.inf or pdf == -1*np.inf):
        return 1.e-10
    return pdf

# gamma CDF is not currently used
def gamma_cdf(x,mean,var):
    if(var < 1.e-10):
        var = 1.e-10
    theta = var / mean
    k = mean / theta
    return  gammainc(k,x/theta) ## wikipedia and scipy define imcomplete gamma differently

################################
# flux to mag conversions

def mag2flux(m, f0 = 1.e6, lim = -10.):
    if(np.min(m) < lim):
        #print "mag below brightness limit; altering"
        m[np.where(m < lim)[0]] = lim
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
