
# if __name__ == '__main__':
#     import matplotlib
#     matplotlib.use('Agg')
#     from matplotlib import rc
#     rc('font',**{'family':'serif','serif':'Computer Modern Roman','size':18})
#     rc('text', usetex=True)


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

def plot_fig(ID, path, nper=2, simulation=False):
    """ plot folded and unfolded LS and censored model fits
    """
    if(simulation):
        data = load_sim(path+ID)
    else:
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
        pfmin = cmodel.optim_fmin(p0,maxiter=1000,ftol=1.,xtol=0.1,mfev=200,fast=True)
        params[(2*i),:] = pfmin
        lliks.append(cmodel.log_likelihood(pfmin))

        # initialized at twice Nat's period
        p0 = cmodel.get_init_par(2 * Pinit[i])
        pfmin = cmodel.optim_fmin(p0,maxiter=1000,ftol=1.,xtol=0.1,mfev=200,fast=True)
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
    fig = plt.gcf()
    fig.set_size_inches(12.0,8.0)
    # unfolded LS
    ax = plt.subplot(221)
    cmodel.plot(ax, orig, fold=False, plot_model=True, mag=True, plot_title = False, period=orig[0])
    ax.set_title("Traditional Method",fontsize=20)
    # unfolded censored
    ax2 = plt.subplot(222)
    cmodel.plot(ax2, pstar, fold=False, plot_model=True, mag=True, plot_title = False, period=pstar[0])
    ax2.set_title("Our Method",fontsize=20)
    # folded LC
    ax3 = plt.subplot(223)
    cmodel.plot(ax3, orig, fold=True, plot_model=True, mag=True, plot_title = False)
    # folded censored
    ax4 = plt.subplot(224)
    cmodel.plot(ax4, pstar, fold=True, plot_model=True, mag=True, plot_title = False)
    
    if(simulation):
        fn = path + 'plots/fig_simulation_'+ ID.split('_')[-1].replace('.','_') +'.png'
    else:
        fn = path + 'plots/fig_'+ ID +'.png'
        
    print 'Writing %s' % fn
    plt.savefig(fn, dpi=200)

    return pstar


def load_sim(fpath):
    """ read in simulated light curve and return dict of t, m, e """
    data = np.loadtxt(fpath, dtype={'names': ('t', 'm', 'e'),
                                    'formats': (np.float, np.float, np.float)})
    return data



if __name__ == '__main__':
    path = '/Users/jwrichar/Documents/CDI/CensoredData/'

    if 0:
        IDs = ['045832-0604.1', '075826-4019.8', '075855-1914.1', '091617-2936.7', '034344+0655.5',\
               '065220+0144.8', '082012-3816.7', '092021-5815.2', '065002-4554.6', '112811-6606.3'  ]

        for ii in IDs:
            plot_fig(ii, path)

        # [-593.81658842874822, -919.90297458282112, -857.12186094575225, -593.06992433308483]

    ID_sims = os.listdir('data/mira_sims')
    for jj in ID_sims:
        plot_fig('data/mira_sims/'+jj, path, simulation=True)
    
