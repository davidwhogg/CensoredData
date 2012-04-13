'''
This file is part of the Censored Data project.
Copyright 2012 David W. Hogg (NYU), Joseph W. Richards, and James Long (UCB)

Should be synced with censoring2.py

issues:
-------
- not tested
- not properly commented
'''

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

def flux2mag(f):
    f0 = 1.e6
    return -2.5 * np.log10(f / f0)

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
    f0 = 1.e-6
    ms = flux2mag(fobss[good])
    ferrs = np.sqrt(sobs2s[good])
    merrs = 0.5 * (flux2mag(fobss[good] - ferrs) - flux2mag(fobss[good] + ferrs))
    return times[good], ms, merrs, times[bad]
