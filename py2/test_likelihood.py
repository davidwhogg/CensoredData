import numpy as np
import likelihood
from matplotlib import pyplot as plt
import matplotlib
from glob import glob
import lomb
from scipy import optimize as op
from multiprocessing import Pool


## CHECK: times2pred and pred2mu
## times2pred and pred2mu look good
t = np.linspace(0,10,1000)
sins, coss = likelihood.times2pred(t,w=2)
A = 4
B = 3
C = 12
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(t,likelihood.pred2mu(sins,coss,A,B,C))
plt.savefig("diag_figs/times2pred_and_pred2mu.pdf")
plt.close()




## CHECK: mag2flux and flux2 mag
t = np.linspace(0,10,1000)
sins, coss = likelihood.times2pred(t,w=.7)
A = 1.5
B = 1.3
C = 7.
mu = likelihood.pred2mu(sins,coss,A,B,C)
f = likelihood.mag2flux(mu)
mu2 = likelihood.flux2mag(f)
# first and third plots should be the same
fig = plt.figure()
ax = fig.add_subplot(311)
ax.plot(t,mu)
ax2 = fig.add_subplot(312)
ax2.plot(t,f)
mu = likelihood.flux2mag(f)
ax3 = fig.add_subplot(313)
ax3.plot(t,mu2)
plt.savefig("diag_figs/mag2flux_and_flux2.pdf")
plt.close()






## CHECK: run model without computing censoring probabilities
## this may return different results than Lomb Scargle
## because least squares are fit in flux space, not mag
## space. however parameter estimtes should look reasonable.
## a plot of the best fit is made
def plot_fits(mira):
    ## get the data
    star = np.loadtxt(mira,
                      usecols=(0,1,2),skiprows=0)
    ctimes = star[star[:,1] > 29.5,0]
    times = star[star[:,1] < 29.5,0]
    mags =  star[star[:,1] < 29.5,1]
    errors = star[star[:,1] < 29.5,2]

    ## get lomb-scargle period
    freqs = lomb.get_freqs2(times)
    rss = lomb.lomb(times,mags,errors,freqs)
    period = 1. / freqs[np.argmin(rss)]
    ## from lomb, get A,B,C values


    sin_c, cos_c = likelihood.times2pred(ctimes,1./period)
    sin_uc, cos_uc = likelihood.times2pred(times,1./period)
    f = likelihood.mag2flux(mags)
    e = likelihood.magerr2fluxerr(mags,errors)
    D = np.array([ np.sin(2*np.pi*times/period), 
                   np.cos(2*np.pi*times/period),
                   np.ones(times.size)])
    A,B,C = np.linalg.lstsq(D.T,mags)[0]


    ## fit censoring model to data
    pars = np.array((A,B,C))
    mu_b = np.min(f)
    v_b = 10.0
    cens = False
    pars_fit = op.fmin(likelihood.nll_fixedB_no_cens,pars,
                       (mu_b,v_b,sin_c,cos_c,sin_uc,cos_uc,f,e))


    ## set x limits to time min and max
    ## add legend 

    ## data for plotting
    times_pred = np.linspace(np.min(times),np.max(times),1000)
    mags_pred_ls = (A*np.sin(2.*np.pi*times_pred / period) 
                    + B*np.cos(2.*np.pi*times_pred / period)
                    + C)
    mags_pred_cens = (pars_fit[0]*np.sin(2.*np.pi*times_pred / period) 
                      + pars_fit[1]*np.cos(2.*np.pi*times_pred / period) 
                      + pars_fit[2])
    xmin = np.min(times)
    xmax = np.max(times)


    fig = plt.figure()
    leg_font = matplotlib.font_manager.FontProperties()
    leg_font.set_size(10)


    ## plot in mag space
    ax = fig.add_subplot(211)
    line1 = ax.plot(times_pred,-mags_pred_ls,color='orange',linewidth=1.5)
    line2 = ax.plot(times_pred,-mags_pred_cens,color='blue',linewidth=1.5)
    ax.plot(ctimes,np.min(-mags)*np.ones(ctimes.size),'o',color='red')
    ax.plot(times,-mags,'o',color="gray",alpha=.5)
    ax.set_yticklabels(np.abs(ax.get_yticks()))
    ax.axis([xmin,xmax,ax.axis()[2],ax.axis()[3]])
    plt.legend((line1,line2),("Lomb-Scargle","Censored"),loc='upper left',
               prop=leg_font)
    ax.set_ylabel("Magnitude")
    ax.set_xlabel("Time")

    ## plot in flux space
    ax = fig.add_subplot(212)
    line1 = ax.plot(times_pred,likelihood.mag2flux(mags_pred_ls),
                    color='orange',linewidth=1.5)
    line2 = ax.plot(times_pred,likelihood.mag2flux(mags_pred_cens),
            color='blue',linewidth=1.5)
    ax.plot(ctimes,likelihood.mag2flux(np.max(mags))*np.ones(ctimes.size),
            'o',color='red')
    ax.plot(times,likelihood.mag2flux(mags),'o',color="gray",alpha=.5)
    ax.axis([xmin,xmax,ax.axis()[2],ax.axis()[3]])
    ax.set_ylabel("Flux")
    ax.set_xlabel("Time")
    plt.legend((line1,line2),("Lomb-Scargle","Censored"),loc='upper left',
               prop=leg_font)
    mira_name = mira.split('/')[-1][:-4]
    plt.savefig("diag_figs/mira_ls_censor_simple/" + mira_name + ".pdf")
    plt.close()


## add gaussian error to flux
mira_periods = "../data/mira_periods.dat"
miras = glob("../data/mira_asas/*")
pool = Pool(processes=7)
pool.map(plot_fits, miras)

## plot uncertainties



## TODO:
## get rid of censored data and try to optimize over A,B,C
## == set mu_b and v_b very low
## 2. from nearby A,B,C
## 3. from far away A,B,C
## 4. make code clean to can repeat 2 and 3 after adding functionality
##    make 2 (or several) nnl functions that optimize different 
##    sets of parameters, one that optimizes just A,B,C and one
##    that optimizes mu_b and v_b as well
## how many iterations, how often convergence (should always converge)
## == change mu_b and v_b, do directions affect likelihood in expected manner
## == let mu_b and v_b vary but have minimums for them (mu_b = 0 min?)
## should get same solutions but mu_b and v_b should go towards minimums
## what are good initial estiamtes of mu_b and v_b










# cens = False
# ll = likelihood.log_likelihood_fixed_w(A,B,C,mu_b,v_b,
#                                        sin_c,cos_c,sin_uc,cos_uc,
#                                        f,e,cens)
# ll

# ll = likelihood.log_likelihood_fixed_w(A+.1,B,C,mu_b,v_b,
#                                        sin_c,cos_c,sin_uc,cos_uc,
#                                        f,e,cens)
# ll


# ll = likelihood.log_likelihood_fixed_w(A+.5,B,C,mu_b,v_b,
#                                        sin_c,cos_c,sin_uc,cos_uc,
#                                        f,e,cens)
# ll


# ll = likelihood.log_likelihood_fixed_w(A+1.,B,C,mu_b,v_b,
#                                        sin_c,cos_c,sin_uc,cos_uc,
#                                        f,e,cens)
# ll



