import numpy as np
import lomb

import glob
from matplotlib import pyplot as plt
from multiprocessing import Pool

def EstimatePeriodAndPlot(ID,
                          f_in="../data/mira_asas/",
                          f_out="diag_figs/mira_plots/"):
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
    miras = glob.glob("../data/mira_asas/*")
    mirasIDs = map(lambda x: x.split("/")[-1].split(".dat")[0],miras)
    pool = Pool(processes=7)
    result = pool.map(EstimatePeriodAndPlot, mirasIDs)
    data = np.column_stack((mirasIDs,result))
    np.savetxt("../data/mira_periods.dat",data,delimiter=" ", fmt="%s")






## effectively need to do a join operation
## we have (periods,miras) and (asas_periods,asas_names) 
## join on miras / asas_names
# periods
# fname = "../data/ACVS.1.1"
# asas_names = np.loadtxt(fname,usecols=(0,),skiprows=2,dtype='S20')
# asas_periods = np.loadtxt(fname,usecols=(1,),skiprows=2)
# mirasIDs = mirasIDs[0:10]
# contained = np.array(map(lambda x: x in mirasIDs,list(asas_names)))





