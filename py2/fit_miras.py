### 
### 
### FIT ALL MIRAS IN data/mira_asas USING ORDINARY LOMB SCARGLE
### THESE WERE SELECTED BY BIG MACC AS LIKELY MIRAS 
###
### by James Long 
### date: TODAY'S DATE 
### 



import numpy as np
import lomb

import glob
from matplotlib import pyplot as plt
from multiprocessing import Pool

def EstimatePeriodAndPlot(ID,
                          f_in="../data/mira_asas/",
                          f_out="diag_figs/mira_plots/"):
    """
    Determine the period and make plot with period of
    light curve with ID.
    """
    print ID
    star = np.loadtxt(f_in + ID + ".dat",
                      usecols=(0,1,2),skiprows=0)
    ctimes = star[star[:,1] > 29.5,0]
    star = star[star[:,1] < 29.5,:]
    cvals = np.array(np.max(star[:,1])) * np.ones(ctimes.shape[0])
    ## estimate period
    freqs = lomb.get_freqs2(star[:,0])
    rss = lomb.lomb(star[:,0],star[:,1],star[:,2],freqs)
    period = 1. / freqs[np.argmin(rss)]
    ## make figure
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(star[:,0],-star[:,1],'o',color="gray",alpha=.5)
    ax.plot(ctimes,-cvals,'ro',alpha=.5)
    ax.set_yticklabels(np.abs(ax.get_yticks()))
    ax.set_xlabel('Time')
    ax.set_ylabel('Magnitude')
    ax2 = fig.add_subplot(212)
    ax2.plot(star[:,0] % period,-star[:,1],'o',color="gray",alpha=.5)
    ax2.plot(ctimes % period,-cvals,'ro',alpha=.5)
    ax2.set_yticklabels(np.abs(ax2.get_yticks()))
    ax2.set_xlabel('Phase')
    ax2.set_ylabel('Magnitude')
    plt.savefig(f_out + ID + ".pdf")
    plt.close()
    return period


if __name__ == "__main__":
    ### first estimate periods for set of miras from ASAS
    ## get filepath/filename.dat
    mira_periods = "../data/mira_periods.dat"

    miras = glob.glob("../data/mira_asas/*")
    ## get filename
    mirasIDs = map(lambda x: x.split("/")[-1].split(".dat")[0],miras)
    ## run EstimatePeriodAndPlot on all mirasIDs in parallel
    pool = Pool(processes=7)
    result = pool.map(EstimatePeriodAndPlot, mirasIDs)
    ## save periods with source id in file
    data = np.column_stack((mirasIDs,result))
    np.savetxt(mira_periods,data,delimiter=" ", fmt="%s")


    ### now make scatterplot of our periods versus ACVS estimated periods
    our = np.loadtxt(mira_periods,usecols=(0,1),
                     dtype=[('ID','S20'),('period',np.float64)])

    ## data/ACVS.1.1 has periods of all asas stars
    fname = '../data/ACVS.1.1'
    acvs = np.loadtxt(fname,usecols=(0,1),
                      dtype=[('ID','S20'),('period',np.float64)])


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


    
