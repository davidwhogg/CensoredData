

import numpy as np
import lomb

import glob
from matplotlib import pyplot as plt


miras = glob.glob("../data/mira_asas/*")
mirasIDs = map(lambda x: x.split("/")[-1].split(".dat")[0],miras)
periods = list()

for ii in range(1000):
    fname = miras[ii]
    star = np.loadtxt(fname,usecols=(0,1,2),skiprows=0)
    star = star[star[:,1] < 29.5,:]
    freqs = lomb.get_freqs2(star[:,0])
    rss = lomb.lomb(star[:,0],star[:,1],star[:,2],freqs)
    print fname
    print 1. / freqs[np.argmin(rss)]
    periods.append(1. / freqs[np.argmin(rss)])
    plt.subplot(211)
    plt.plot(star[:,0],-star[:,1],'ro')
    plt.subplot(212)
    plt.plot(star[:,0] % periods[-1],-star[:,1],'ro')
    plt.savefig("diag_figs/mira_plots/" + mirasIDs[ii] + ".pdf")
    plt.close()




## effectively need to do a join operation
## we have (periods,miras) and (asas_periods,asas_names) 
## join on miras / asas_names
periods
fname = "../data/ACVS.1.1"
asas_names = np.loadtxt(fname,usecols=(0,),skiprows=2,dtype='S20')
asas_periods = np.loadtxt(fname,usecols=(1,),skiprows=2)
mirasIDs = mirasIDs[0:10]
contained = np.array(map(lambda x: x in mirasIDs,list(asas_names)))




