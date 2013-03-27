"""
This file is part of the Censored Data Project
Copyright 2012, Joseph W. Richards, David W. Hogg, James Long

This code analyzes all ASAS Mira objects using parallel processing

Issues:

"""

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import rc
    rc('font',**{'family':'serif','serif':'Computer Modern Roman','size':18})
    rc('text', usetex=True)


from multiprocessing import Pool
import censoring2 as c2
import cProfile
import numpy as np
import matplotlib.pylab as plt
import sys, os
import time
import urllib2 as ulib

from fit_censoring import *

#global nper
nper = 2

#global path
path = '/Users/jwrichar/Documents/CDI/CensoredData/'

#global catalog
cat_data = np.loadtxt(path + 'data/asas_class_catalog_v2_3.dat',\
                     usecols=(0,1,4,5,6,9,37), skiprows=1,\
                     dtype=[('ID','S20'), ('dID','i4'), ('class','S20'), ('Pclass',np.float), \
                            ('anom',np.float64), ('Pmira',np.float), ('P','f16')], delimiter=',')


def doMira(ind, catalog):
    """
    analyze a single ASAS mira variable.  This method is meant to be used with the
    multiprocessing module to analyze the entire ASAS Mira data set
    """

    print '   #### doing mira: ' + str(catalog['ID'][ind]) +\
          ' dotAstro: ' + str(catalog['dID'][ind])

    f = open(path + 'doMira.log', 'a')
    f.write( 'starting mira: ' + str(catalog['ID'][ind]) +\
          ' dotAstro: ' + str(catalog['dID'][ind]) +'\n')
    f.close()

    new_periods = np.zeros((1,),dtype=[('ID','S16'), ('oldP',np.float), ('oldA',np.float),\
                                       ('newP',np.float), ('newA',np.float),('llik',np.float)])

    tryind = 0
    while(tryind < 5):
        try:
            data = load_lc_from_web(catalog['ID'][ind])
            break
        except:
            tryind += 1
            print 'Cannot retreive data from web'
            time.sleep(10)

    if(tryind >= 5):
        print 'Skipping Mira ' + str(catalog['dID'][ind])
        return
    
    indo = np.where(data['m'] != 29.999)
    tobs = data['t'][indo]
    mobs = data['m'][indo]
    eobs = data['e'][indo]
    tcen = data['t'][np.where(data['m'] == 29.999)]

    # original L-S period (w/o using censored data)
    Porig = periods_lomb(tobs, mobs, eobs, n_per = 1)[0]
    new_periods[0][0] = catalog['ID'][ind]
    new_periods[0][1] = Porig

    # instantiate Censor object
    cmodel = c2.Censored(tobs, mobs, eobs, tcen, name = catalog['ID'][ind], tolquad=1.)

    # get original amplitude
    orig = cmodel.get_init_par(Porig)
    new_periods[0][2] = np.sqrt(orig[2]**2 + orig[3]**2)

    # do Lomb-scargle search to get top few periods
    Pinit = period_lombc(tobs, mobs, eobs, tcen, n_per = nper,df=1e-5)
    p0 = cmodel.get_init_par(Pinit[0])
    #ll0 = cmodel.log_likelihood(p0, fast=True)

    #print 'Log-likelihood at p0: ' + str(ll0)
    #if np.isnan(ll0):
    #    raise Exception

    lliks = []
    params = np.zeros((2*nper,10))
    for i in np.arange(nper):
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
    new_periods[0][3] = pstar[0]
    new_periods[0][4] = np.sqrt(pstar[2]**2 + pstar[3]**2)
    new_periods[0][5] = np.max(lliks)
    
    print 'New Period: ' + str(round(pstar[0],2))
    print 'Old Period: ' + str(round(Porig,3))
        
    # save result as line in txt file
#    np.savetxt(path + 'plots/bestpar_'+ catalog['ID'][ind] + '.dat', pstar)
    
    # plot folded model with new method
    plt.clf()
    ax = plt.subplot(111)
    cmodel.plot(ax, pstar, fold=True)
    plt.savefig(path + 'plots/bestfit_'+ catalog['ID'][ind] +'_new.png')
    # plot in magnitudes
    plt.clf()
    ax = plt.subplot(111)
    cmodel.plot(ax, pstar, fold=True, mag=True)
    plt.savefig(path + 'plots/bestfit_mag_'+ catalog['ID'][ind] +'_new.png')
    
    # plot folded model with old method
    pold = pstar
    pold[0] = Porig
    plt.clf()
    ax = plt.subplot(111)
    cmodel.plot(ax, pold, fold=True, plot_model = False)
    plt.savefig(path + 'plots/bestfit_'+ catalog['ID'][ind] +'_old.png')
    # plot in mags
    plt.clf()
    ax = plt.subplot(111)
    cmodel.plot(ax, pold, fold=True, plot_model = False, mag = True)
    plt.savefig(path + 'plots/bestfit_mag_'+ catalog['ID'][ind] +'_old.png')
    
    f_handle = file(path + 'data/new_periods.dat', 'a')
    np.savetxt(f_handle, new_periods, delimiter = ',',fmt = '%s %s %s %s %s %s')
    f_handle.close()
    
    return catalog['ID'][ind]

def doMira_partial(ind):
    return doMira(ind, cat_data)

if __name__ == '__main__':

    p_mira = 0.75
    miras = np.where(np.logical_and(cat_data['Pmira'] > p_mira , cat_data['anom'] < 3.))[0] 

    pool = Pool(processes=2)

    result = pool.map(doMira_partial, miras)
    pool.close()
    pool.join()
