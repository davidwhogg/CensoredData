## code to test nat's function
## by James Long on April 23, 2012

import numpy as np
from lomb_scargle_censor import lomb

# get data in nat's preferred form
i = "../data/lc_085647-6242.9.dat"
lc = np.fromfile(i, dtype=float, count=-1, sep=" ")
lc = lc.reshape((lc.size / 3,3))
time_missing = lc[lc[:,1] > 29.9,0]
lc = lc[lc[:,1] < 29.99,:]

# l.s. grid search parameters
min_freq = 1./1000.
max_freq = 1./50.
df = .000001
numf = int((max_freq - min_freq) / df)

## best period using censored data
results = lomb(lc[:,0],lc[:,1],lc[:,2],time_missing,min_freq,df,numf)
best = np.argmax(results)
period = 1 / (min_freq + best * df)
print period

## best period ignoring censoring
none_missing = np.array([],dtype='float64')
results = lomb(lc[:,0],lc[:,1],lc[:,2],none_missing,min_freq,df,numf)
best = np.argmax(results)
period = 1 / (min_freq + best * df)
print period
