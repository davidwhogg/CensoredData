

import censoring2 as c2
import cProfile
import numpy as np
import matplotlib.pylab as plt
import sys, os
import urllib2 as ulib

sys.path.append(os.path.abspath(os.environ.get("TCP_DIR") + \
                                      '/Algorithms/fitcurve'))
from lomb_scargle_refine import lomb as lombr
from lomb_scargle_censor import lomb as lombc

path = '/Users/jwrichar/Documents/CDI/CensoredData/'

def periods_lomb(t, f, e, n_per=3,f0=1./1000,df=1e-5,fn=5.):

    numf = int((fn-f0)/df)
    ftest = 1.*f
    P_cand = []
    for i in xrange(n_per):
        # lomb-scargle periodogram
        psd,res = lombr(t,ftest,e,f0,df,numf, detrend_order=1,nharm=1)
        P_cand.append(1./res['freq'])
        ftest -= res['model']

    return P_cand


def period_lombc(t, f, e, tc,f0=1./1000,df=1e-5,fn=5., n_per = 5):

    numf = int((fn-f0)/df)
    freqgrid = np.linspace(f0,fn,numf)
    ftest = 1.*f
    # lomb-scargle periodogram
    psd = lombc(t,ftest,e,tc,f0,df,numf)

    #plt.plot(freqgrid,psd)
    #plt.show()
    
    plt.plot(1./freqgrid,psd)
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
    miradata = np.loadtxt(ur.readlines(), usecols=(0,3,8,11), \
                          dtype=[('t',np.float64), ('m',np.float64), ('e',np.float64),\
                                 ('grade','S2')])
    use = np.where(np.logical_and(np.logical_and(miradata['grade'] != 'D',miradata['grade'] != 'F'),\
                                  miradata['m'] != 99.999))[0]
    return miradata[use,:]


if __name__ == "__main__":

    # load catalog
    catalog = np.loadtxt('/Users/jwrichar/Documents/CDI/ASAS/ASASCatalog/catalog/asas_class_catalog_v2_3.dat',\
                         usecols=(0,1,4,5,6,9,37), skiprows=1,\
                         dtype=[('ID','S20'), ('dID','i4'), ('class','S20'), ('Pclass',np.float), \
                                ('anom',np.float64), ('Pmira',np.float), ('P','f16')], delimiter=',')


    # select all miras (P(Mira) > 0.75 and anom_score < 3.0)
    miras = np.where(np.logical_and(catalog['Pmira'] > 0.75 , catalog['anom'] < 3.))[0] # 2538 mira variables
    

    #### FOR EACH MIRA IN miras ####
    for jj in np.arange(100):
        # load in data from web
        print 'doing mira ' + str(catalog['ID'][miras[jj]]) + ' dotAstro: ' + str(catalog['dID'][miras[jj]])
    
        data = load_lc_from_web(catalog['ID'][miras[jj]])
        
        indo = np.where(data['m'] != 29.999)
        tobs = data['t'][indo]
        mobs = data['m'][indo]
        eobs = data['e'][indo]
        tcen = data['t'][np.where(data['m'] == 29.999)]

        # instantiate Censor object
        cmodel = c2.Censored(tobs, mobs, eobs, tcen, name = catalog['ID'][miras[jj]])

        # do Lomb-scargle search to get top few periods
        nper = 1 # number of periods to initialize over
        Pinit = period_lombc(tobs, mobs, eobs, tcen, n_per = nper,df=1e-5)
        #    Ps = periods_lomb(tobs, mobs, eobs)
        print Pinit

        p0 = cmodel.get_init_par(Pinit[0])
        ll0 = cmodel.log_likelihood(p0)
        if np.isnan(ll0):
            continue
        print 'Log-likelihood at p0: ' + str(ll0)

        lliks = []
        params = np.zeros((nper,10))
        for i in np.arange(nper):
            # initial guess for model parameters
            # params:     P,A0,A1,B1,su2,B,VB,Vsig,S,VS
            p0 = cmodel.get_init_par(Pinit[i])
            print p0
            pfmin = cmodel.optim_fmin(p0,maxiter=1000,ftol=1.,xtol=0.1,mfev=200)
            print pfmin
            params[i,:] = pfmin
            lliks.append(cmodel.log_likelihood(pfmin))

        # optimal parameter vector
        pstar = params[np.argmax(lliks),:]
        print pstar

        # save result as line in txt file
        np.savetxt(path + 'plots/bestpar_'+ catalog['ID'][miras[jj]] + '.dat', pstar)

        # plot folded model with new / old method
        plt.clf()
        ax = plt.subplot(111)
        cmodel.plot(ax, pstar, fold=True)
        plt.savefig(path + 'plots/bestfit_'+ catalog['ID'][miras[jj]] +'_new.png')

        pold = pstar
        pold[0] = catalog['P'][miras[jj]]
        plt.clf()
        ax = plt.subplot(111)
        cmodel.plot(ax, pstar, fold=True, plot_model = False)
        plt.savefig(path + 'plots/bestfit_'+ catalog['ID'][miras[jj]] +'_old.png')

    




    


