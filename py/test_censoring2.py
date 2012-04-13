
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import rc
    rc('font',**{'family':'serif','serif':'Computer Modern Roman','size':18})
    rc('text', usetex=True)

import censoring2 as c2
import cProfile
import scipy.stats as stats
import numpy as np
import scipy.optimize as op
import matplotlib.pylab as plt


def plot_llmarg(cens, pstar, ind=0, p0=0, pn=1, n=11, name='P'):

    ll_p = []
    true = pstar[ind]
    parval = np.linspace(p0,pn,n)
    for pp in parval:
        ptmp = pstar + []; ptmp[ind] = pp
        ll_p.append(cens.log_likelihood(ptmp))
    plt.plot(parval, ll_p, '-o')
    plt.xlabel(name)
    plt.ylabel('log likelihood')
    plt.text(0.5, 0.1, 'True value = ' + str(round(true,3)), transform = ax.transAxes)
    plt.show()


if __name__ == "__main__":

    np.random.seed(47)

    import fake_data as fd
    
    A0 = 100.; A1 = 50.;    B1 = 50.
    P = 400.
    w = (2. * np.pi) / P
    s2mu = 25.**2
    B, VB = 100., 20.**2
    Vsigma = 5.**2
    S, VS = 25., 5.**2

    gt, m, mer, bt = fd.make_fake_data(3600. * np.random.uniform(size=512), \
                                      w, A0, A1, B1, s2mu, B, VB, Vsigma, S, VS)
    
    cmodel = c2.Censored(gt,m,mer,bt,tolquad=1.)
    popt = [P,A0,A1,B1,s2mu,B,VB,Vsigma,S,VS]

    p0 = cmodel.get_init_par(400.)
    print p0

    plt.clf()
    ax = plt.subplot(111)
    cmodel.plot(ax,p0)
    plt.savefig('init_params.png')

    plt.clf()
    ax = plt.subplot(111)
    cmodel.plot(ax,p0, fold=True)
    plt.savefig('init_params_folded.png')

    plt.clf()
    ax = plt.subplot(111)
    cmodel.plot(ax,popt)
    plt.savefig('true_params.png')

    plt.clf()
    ax = plt.subplot(111)
    cmodel.plot(ax,popt, fold=True)
    plt.savefig('true_params_folded.png')


    prof = True
    if(prof): # profile code?
        cProfile.run("cmodel.log_likelihood(popt)")


    plotmarg = False

    if(plotmarg):
        # make a plot of ll vs. parameter
        fig = plt.figure(figsize=(16, 10))
        fig.subplots_adjust(hspace=0.7)
        fig.subplots_adjust(wspace=0.2)

        ax = fig.add_subplot(5,2,1)
        plot_llmarg(cmodel,popt,0,p0=popt[0]*0.9, pn = popt[0]*1.1, n=11, name='Period')

        ax = fig.add_subplot(5,2,2)
        plot_llmarg(cmodel,popt,1,p0=popt[1]*0.9, pn = popt[1]*1.1, n=11, name='A0')

        ax = fig.add_subplot(5,2,3)
        plot_llmarg(cmodel,popt,2,p0=popt[2]*0.9, pn = popt[2]*1.1, n=11, name='A1')

        ax = fig.add_subplot(5,2,4)
        plot_llmarg(cmodel,popt,3,p0=popt[3]*0.9, pn = popt[3]*1.1, n=11, name='B1')

        ax = fig.add_subplot(5,2,5)
        plot_llmarg(cmodel,popt,4,p0=popt[4]*0.1, pn = popt[4]*1.9, n=11, name=r'$s_{\mu}^2$')

        ax = fig.add_subplot(5,2,6)
        plot_llmarg(cmodel,popt,7,p0=popt[7]*0.1, pn = popt[7]*1.9, n=11, name=r'$V_{\sigma}$')

        ax = fig.add_subplot(5,2,7)
        plot_llmarg(cmodel,popt,5,p0=popt[5]*0.9, pn = popt[5]*1.1, n=11, name='B')

        ax = fig.add_subplot(5,2,8)
        plot_llmarg(cmodel,popt,6,p0=popt[6]*0.1, pn = popt[6]*1.9, n=11, name=r'$V_B$')

        ax = fig.add_subplot(5,2,9)
        plot_llmarg(cmodel,popt,8,p0=popt[8]*0.9, pn = popt[8]*1.1, n=11, name='S')

        ax = fig.add_subplot(5,2,10)
        plot_llmarg(cmodel,popt,9,p0=popt[9]*0.1, pn = popt[9]*1.9, n=11, name=r'$V_S$')
        plt.savefig('likelihood_marg.png')



#[400.0, 100.0, 50.0, 50.0, 625.0, 100.0, 400.0, 100.0, 25.0, 25.0]
#[ 400.          117.04071511   35.79569983   34.50307783  547.94115976
#  121.10564877  121.10564877   23.00652436   23.00652436  529.30016307]

    maxlik = True
    if(maxlik):
        #p0 = popt
        #p0 = popt + stats.norm.rvs(0,.01,size=10)
        print p0
        pfmin = cmodel.optim_fmin(p0,maxiter=1000,ftol=0.5,xtol=0.1,mfev=500)
        print cmodel.log_likelihood(pfmin)
        print cmodel.log_likelihood(p0)

        plt.clf()
        ax = plt.subplot(111)
        cmodel.plot(ax,pfmin)
        plt.savefig('bestfit_params.png')

        plt.clf()
        ax = plt.subplot(111)
        cmodel.plot(ax,pfmin, fold=True)
        plt.savefig('bestfit_params_folded.png')

        print pfmin
        print popt



# params:     P,A0,A1,B1,su2,B,VB,Vsig,S,VS





# read in a mira LC and fmin that baby



# see if the solution is sensible

