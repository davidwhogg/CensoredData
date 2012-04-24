
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

sys.path.append(os.path.abspath(os.environ.get("TCP_DIR") + \
                                      '/Algorithms/fitcurve'))
from lomb_scargle import lomb as lomb
from lomb_scargle_refine import lomb as lombr
from lomb_scargle_censor import lomb as lombc

path = '/Users/jwrichar/Documents/CDI/CensoredData/'

def periods_lomb(t, f, e, n_per=3,f0=1./1000,df=1e-5,fn=1./50.,plot=False):

    numf = int((fn-f0)/df)
    freqgrid = np.linspace(f0,fn,numf)
    ftest = 1.*f
    P_cand = []
    for i in xrange(n_per):
        psd,res = lombr(t,ftest,e,f0,df,numf, detrend_order=1,nharm=8)
        if(plot):
            plt.plot(freqgrid,psd)
            plt.show()

        P_cand.append(1./res['freq'])
        ftest -= res['model']
    return P_cand


def period_lombc(t, f, e, tc,f0=1./1000,df=1e-5,fn=1./50., n_per = 5, plot=False):

    numf = int((fn-f0)/df)
    freqgrid = np.linspace(f0,fn,numf)
    ftest = 1.*f
    # lomb-scargle periodogram
    psd = lombc(t,ftest,e,tc,f0,df,numf)
    if(plot):
        plt.plot(freqgrid,psd)
        plt.show()

    indpeak = find_peaks(psd) # find peaks in LS
    indtop = np.where(psd[indpeak] >= np.sort(psd[indpeak])[-n_per])[0] 
    
    P_cand = (1./freqgrid[indpeak[indtop]]) # top n_per periods
    return P_cand

def find_peaks(x):
    xmid = x[1:-1] # orig array with ends removed
    xm1 = x[2:] # orig array shifted one up
    xp1 = x[:-2] # origh array shifted one back
    return np.where(np.logical_and(xmid > xm1, xmid > xp1))[0] + 1

def load_lc_from_web(ID):
    urlpath = 'http://www.astrouw.edu.pl/cgi-asas/asas_cgi_get_data?'
    ur = ulib.urlopen( urlpath + ID + ',asas3')#open url
    miradata = np.loadtxt(ur.readlines(), usecols=(0,2,7,11), \
                          dtype=[('t',np.float64), ('m',np.float64), ('e',np.float64),\
                                 ('grade','S2')])
    use = np.where(np.logical_and(np.logical_and(miradata['grade'] != 'D',miradata['grade'] != 'F'),\
                                  miradata['m'] != 99.999))[0]
    return miradata[use,:]


def catchnan(p):
    ll0 = cmodel.log_likelihood(p)
    if(np.isnan(ll0)):
        1./0
    pass

if __name__ == "__main__":

    print '######### Fitting Miras ###########'
    
    # load catalog
    catalog = np.loadtxt(path + 'data/asas_class_catalog_v2_3.dat',\
                         usecols=(0,1,4,5,6,9,37), skiprows=1,\
                         dtype=[('ID','S20'), ('dID','i4'), ('class','S20'), ('Pclass',np.float), \
                                ('anom',np.float64), ('Pmira',np.float), ('P','f16')], delimiter=',')


    # select all miras (P(Mira) > 0.75 and anom_score < 3.0)
    miras = np.where(np.logical_and(catalog['Pmira'] > 0.75 , catalog['anom'] < 3.))[0] # 2538 mira variables

#    new_periods = np.zeros((len(miras),6),dtype=[('ID','S10'), ('oldP',np.float), ('oldA',np.float),\
#                                                 ('newP',np.float), ('newA',np.float),('llik',np.float)])

    #### FOR EACH MIRA IN miras ####
    for jj in np.arange(0,10):
    #for jj in np.arange(486,487):
        # load in data from web
        print '   #### doing mira ' + str(jj) + ': ' + str(catalog['ID'][miras[jj]]) \
              + ' dotAstro: ' + str(catalog['dID'][miras[jj]])

        # initialize new period vector
        new_periods = np.zeros((1,),dtype=[('ID','S16'), ('oldP',np.float), ('oldA',np.float),\
                                        ('newP',np.float), ('newA',np.float),('llik',np.float)])

        print new_periods
        
        data = load_lc_from_web(catalog['ID'][miras[jj]])
        
        indo = np.where(data['m'] != 29.999)
        tobs = data['t'][indo]
        mobs = data['m'][indo]
        eobs = data['e'][indo]
        tcen = data['t'][np.where(data['m'] == 29.999)]

        # original L-S period (w/o using censored data)
        Porig = periods_lomb(tobs, mobs, eobs, n_per = 1)[0]
        new_periods[0][0] = catalog['ID'][miras[jj]]
        new_periods[0][1] = Porig

        # instantiate Censor object
        cmodel = c2.Censored(tobs, mobs, eobs, tcen, name = catalog['ID'][miras[jj]], tolquad=1.)

        # get original amplitude
        orig = cmodel.get_init_par(Porig)
        new_periods[0][2] = np.sqrt(orig[2]**2 + orig[3]**2)

        # do Lomb-scargle search to get top few periods
        nper = 2 # number of periods to initialize over
        Pinit = period_lombc(tobs, mobs, eobs, tcen, n_per = nper,df=1e-5)

        p0 = cmodel.get_init_par(Pinit[0])
        ll0 = cmodel.log_likelihood(p0, fast=True)
        print 'Log-likelihood at p0: ' + str(ll0)
        if np.isnan(ll0):
            raise Exception
            #continue

        lliks = []
        params = np.zeros((2*nper,10))
        for i in np.arange(nper):
            # initial guess for model parameters
            # params:     P,A0,A1,B1,su2,B,VB,Vsig,S,VS

            # initialized at Nat's period
            p0 = cmodel.get_init_par(Pinit[i])
            pfmin = cmodel.optim_fmin(p0,maxiter=1000,ftol=1.,xtol=0.1,mfev=500,fast=True)
            #print pfmin
            params[(2*i),:] = pfmin
            lliks.append(cmodel.log_likelihood(pfmin))

            # initialized at twice Nat's period
            p0 = cmodel.get_init_par(2 * Pinit[i])
            pfmin = cmodel.optim_fmin(p0,maxiter=1000,ftol=1.,xtol=0.1,mfev=500,fast=True)
            params[((2*i)+1),:] = pfmin
            lliks.append(cmodel.log_likelihood(pfmin))


        print lliks
        print 'Optimized log-likelihood: ' + str(np.max(lliks))
        # optimal parameter vector
        pstar = params[np.argmax(lliks),:]
        #print pstar
        new_periods[0][3] = pstar[0]
        new_periods[0][4] = np.sqrt(pstar[2]**2 + pstar[3]**2)
        new_periods[0][5] = np.max(lliks)

        print 'New Period: ' + str(round(pstar[0],2))
        print 'Old Period: ' + str(round(Porig,3))

        print jj
        print new_periods
        
        # save result as line in txt file
        np.savetxt(path + 'plots/bestpar_'+ catalog['ID'][miras[jj]] + '.dat', pstar)


        #####  PLOTS  #######
        # plot folded model with new method
        plt.clf()
        ax = plt.subplot(111)
        cmodel.plot(ax, pstar, fold=True)
        plt.savefig(path + 'plots/bestfit_'+ catalog['ID'][miras[jj]] +'_new.png')
        # plot in magnitudes
        plt.clf()
        ax = plt.subplot(111)
        cmodel.plot(ax, pstar, fold=True, mag=True)
        plt.savefig(path + 'plots/bestfit_mag_'+ catalog['ID'][miras[jj]] +'_new.png')

        # plot folded model with old method
        pold = pstar
        pold[0] = Porig
        plt.clf()
        ax = plt.subplot(111)
        cmodel.plot(ax, pold, fold=True, plot_model = False)
        plt.savefig(path + 'plots/bestfit_'+ catalog['ID'][miras[jj]] +'_old.png')
        # plot in mags
        pold = pstar
        pold[0] = catalog['P'][miras[jj]]
        plt.clf()
        ax = plt.subplot(111)
        cmodel.plot(ax, pold, fold=True, plot_model = False, mag = True)
        plt.savefig(path + 'plots/bestfit_mag_'+ catalog['ID'][miras[jj]] +'_old.png')


        f_handle = file(path + 'data/new_periods.dat', 'a')
        np.savetxt(f_handle, new_periods, delimiter = ',',fmt = '%s %s %s %s %s %s')
        f_handle.close()




