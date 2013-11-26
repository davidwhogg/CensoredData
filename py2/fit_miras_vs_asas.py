### 
### 
### COMPARE PERIODS FIT BY OUR L-S TO ASAS TEAM 
###  
###
### by Long, Richards, and Hogg
### date: Nov 26, 2013
### 

import numpy as np
from matplotlib import pyplot as plt

## data/mira_periods.dat has mira periods
## estimated by py2/fit_miras.py
fname = '../data/mira_periods.dat'
our = np.loadtxt(fname,usecols=(0,1),dtype=[('ID','S20'),('period',np.float64)])


## data/ACVS.1.1 has periods of all asas stars
fname = '../data/ACVS.1.1'
acvs = np.loadtxt(fname,usecols=(0,1),dtype=[('ID','S20'),('period',np.float64)])


present = np.array(map(lambda x: x in our['ID'],acvs['ID']))
acvs = acvs[present]


## now sort both our and acvs by star name
our.sort(order='ID')
acvs.sort(order='ID')


## often get same period, sometimes off by multiple of 2
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(our['period'],acvs['period'],'o',color="gray",alpha=.5)
ax.set_xlabel('Our LS')
ax.set_ylabel('ACVS LS')
plt.savefig("diag_figs/ourLS_versus_ACVSls.pdf")
plt.close()

