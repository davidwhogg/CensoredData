
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

    import fake_data as fd
    
    A0 = 5.; A1 = 1.;    B1 = 1.
    P = 400.
    w = (2 * np.pi) / P
    s2mu = 0.2**2
    S, VS = 0.7, 0.15**2
    B, VB = 4.5, 0.4**2
    Vsigma = 0.2**2

    gt, f, s2, bt = fd.make_fake_data(3600. * np.random.uniform(size=250), \
                                      w, A0, A1, B1, s2mu, B, VB, Vsigma, S, VS)

    cmodel = c2.Censored(gt,f,s2,bt,tolquad=1.)
    popt = [P,A0,A1,B1,s2mu,B,VB,Vsigma,S,VS]

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
        plot_llmarg(cmodel,popt,0,p0=350, pn = 450, n=11, name='Period')

        ax = fig.add_subplot(5,2,2)
        plot_llmarg(cmodel,popt,1,p0=2, pn = 8, n=11, name='A0')

        ax = fig.add_subplot(5,2,3)
        plot_llmarg(cmodel,popt,2,p0=0.5, pn = 1.5, n=11, name='A1')

        ax = fig.add_subplot(5,2,4)
        plot_llmarg(cmodel,popt,3,p0=0.5, pn = 1.5, n=11, name='B1')

        ax = fig.add_subplot(5,2,5)
        plot_llmarg(cmodel,popt,4,p0=0.01, pn = 0.07, n=11, name=r'$s_{\mu}^2$')

        ax = fig.add_subplot(5,2,6)
        plot_llmarg(cmodel,popt,5,p0=3, pn = 6, n=11, name='B')

        ax = fig.add_subplot(5,2,7)
        plot_llmarg(cmodel,popt,6,p0=0.1, pn = 0.22, n=11, name=r'$V_B$')

        ax = fig.add_subplot(5,2,8)
        plot_llmarg(cmodel,popt,7,p0=0.01, pn = 0.07, n=11, name=r'$V_{\sigma}$')

        ax = fig.add_subplot(5,2,9)
        plot_llmarg(cmodel,popt,8,p0=0.4, pn = 1.0, n=11, name='S')

        ax = fig.add_subplot(5,2,10)
        plot_llmarg(cmodel,popt,9,p0=0.01, pn = 0.035, n=11, name=r'$V_S$')
        plt.savefig('likelihood_marg.png')


    maxlik = False
    if(maxlik):
        p0 = popt
        #p0 = popt + stats.norm.rvs(0,.01,size=10)
        pfmin = cmodel.optim_fmin(p0,maxiter=1000,ftol=0.5,xtol=0.1)
        #pfmin = cmodel.optim_fmin(p0,maxiter=100,ftol=0.01)
        print pfmin
        print popt



# params:     P,A0,A1,B1,su2,B,VB,Vsig,S,VS





# read in a mira LC and fmin that baby



# see if the solution is sensible

