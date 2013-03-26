
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import rc
    rc('font',**{'family':'serif','serif':'Computer Modern Roman','size':18})
    rc('text', usetex=True)


import censoring2 as c2
import cProfile
import numpy as np
import matplotlib.pylab as plt
import sys, os
import urllib2 as ulib

import pdb

from lomb_scargle import lomb as lomb
from lomb_scargle_refine import lomb as lombr
from lomb_scargle_censor import lomb as lombc

from fit_censoring import load_lc_from_web, periods_lomb, period_lombc

def plot_fig(ID, path, nper=2):
    """ plot folded and unfolded LS and censored model fits
    """
    data = load_lc_from_web(ID)

    indo = np.where(data['m'] != 29.999)
    tobs = data['t'][indo]
    mobs = data['m'][indo]
    eobs = data['e'][indo]
    tcen = data['t'][np.where(data['m'] == 29.999)]

    # original L-S period (w/o using censored data)
    Porig = periods_lomb(tobs, mobs, eobs, n_per = 1)[0]

    # instantiate Censor object
    cmodel = c2.Censored(tobs, mobs, eobs, tcen, name = ID, tolquad=1.)

    # get LS amplitude
    orig = cmodel.get_init_par(Porig)
    print orig
    # LS amplitude new_periods[0][2] = np.sqrt(orig[2]**2 + orig[3]**2)

    # do Lomb-scargle search to get top few periods
    Pinit = period_lombc(tobs, mobs, eobs, tcen, n_per = nper,df=1e-5)
    p0 = cmodel.get_init_par(Pinit[0])
    print p0
    #ll0 = cmodel.log_likelihood(p0, fast=True)

    #print 'Log-likelihood at p0: ' + str(ll0)
    #if np.isnan(ll0):
    #    raise Exception

    lliks = []
    params = np.zeros((2*nper,10))
    for i in np.arange(nper):
        print "Fitting for period candidate %i of %i" % (i, nper)
        # initialized at Nat's period
        p0 = cmodel.get_init_par(Pinit[i])
        pfmin = cmodel.optim_fmin(p0,maxiter=1000,ftol=1.,xtol=0.1,mfev=1000,fast=True)
        params[(2*i),:] = pfmin
        lliks.append(cmodel.log_likelihood(pfmin))

        # initialized at twice Nat's period
        p0 = cmodel.get_init_par(2 * Pinit[i])
        pfmin = cmodel.optim_fmin(p0,maxiter=1000,ftol=1.,xtol=0.1,mfev=1000,fast=True)
        params[((2*i)+1),:] = pfmin
        lliks.append(cmodel.log_likelihood(pfmin))

    print lliks
    print 'Optimized log-likelihood: ' + str(np.nanmax(lliks))
    # optimal parameter vector
    pstar = params[np.where(lliks==np.nanmax(lliks))[0],:][0]
    #new_periods[0][3] = pstar[0] # new period
    #new_periods[0][4] = np.sqrt(pstar[2]**2 + pstar[3]**2)
    #new_periods[0][5] = np.max(lliks)
    
    print 'New Period: ' + str(round(pstar[0],3))
    print 'Old Period: ' + str(round(Porig,3))

  
    # make plots
    plt.clf()
    # unfolded LS
    ax = plt.subplot(221)
    cmodel.plot(ax, orig, fold=False, plot_model=True, mag=True)
    # unfolded censored
    ax = plt.subplot(222)
    cmodel.plot(ax, pstar, fold=False, plot_model=True, mag=True)
    # folded LC
    ax = plt.subplot(223)
    cmodel.plot(ax, orig, fold=True, plot_model=True, mag=True)
    # folded censored
    ax = plt.subplot(224)
    cmodel.plot(ax, pstar, fold=True, plot_model=True, mag=True)
    plt.savefig(path + 'plots/fig_'+ ID +'.png')

    return pstar


if __name__ == '__main__':
    # choose a mira to run it on
    path = '/Users/jwrichar/Documents/CDI/CensoredData/'
    ID = '045832-0604.1'

    plot_fig(ID,path)
