'''
This file is part of the Censored Data project.
Copyright 2012 David W. Hogg (NYU).

issues:
-------
- not tested
- not properly commented
'''

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import rc
    rc('font',**{'family':'serif','serif':'Computer Modern Roman','size':12})
    rc('text', usetex=True)
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm as Norm
from scipy.stats import gamma as Gamma

def mu(times, omega, A0, A1, B1):
    '''
    Make the model mean flux given lightcurve parameters.
    '''
    return A0 \
        + A1 * np.cos(omega * times) \
        + B1 * np.sin(omega * times)

def make_fake_data(times, omega, A0, A1, B1, s2mu, B, VB, Vsigma, S, VS):
    '''
    Implement the full generative model from the paper.  Is this
    correct?
    '''
    mus = mu(times, omega, A0, A1, B1)
    sigma2s = Gamma.rvs(S**2/VS, scale=VS/S, size=times.size)
    sobs2s = np.array([Gamma.rvs(m**2/Vsigma, scale=Vsigma/m, size=1)[0] for m in sigma2s])
    fobss = mus + np.sqrt(sigma2s + s2mu) * Norm.rvs(size=times.size)
    bs = B + np.sqrt(VB) * Norm.rvs(size=times.size)
    good = fobss > bs
    bad = good == False
    return times[good], fobss[good], sobs2s[good], times[bad]

def plot_data(ax, goodtimes, fobss, sobs2s, badtimes):
    ax.errorbar(goodtimes, fobss, fmt='ko', yerr=np.sqrt(sobs2s))
    ax.plot(badtimes, np.zeros_like(badtimes), 'ro')
    return None

def plot_model(ax, omega, A0, A1, B1):
    tlim = ax.get_xlim()
    tp = np.linspace(tlim[0], tlim[1], 10000)
    fp = mu(tp, omega, A0, A1, B1)
    ax.plot(tp, fp, 'k-', alpha=0.5)
    return None

if __name__ == '__main__':
    omega, A0, A1, B1 = 2. * np.pi / 400., 5.0, 3.0, 0.5
    s2mu = 0.2**2
    S, VS = 0.7, 0.15**2
    B, VB = 4.5, 0.4**2
    Vsigma = 0.2**2
    gt, f, s2, bt = make_fake_data(3600. * np.random.uniform(size=512), omega, A0, A1, B1, s2mu, B, VB, Vsigma, S, VS)
    ax = plt.subplot(111)
    plot_data(ax, gt, f, s2, bt)
    plot_model(ax, omega, A0, A1, B1)
    plt.xlabel(r'time $t$')
    plt.ylabel(r'flux $f$')
    plt.savefig('whatever.png')
