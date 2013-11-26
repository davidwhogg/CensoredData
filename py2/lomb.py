### 
### 
### PYTHON IMPLEMENTATION OF LOMB SCARGLE 
###   
###
### by James Long 
### date: OCTOBER 23, 2013 
### 



import numpy as np




def lomb(times,mags,mags_errors,freqs):
    rss = []
    for freq in freqs:
        A = np.array([ np.sin(2*np.pi*times*freq), np.cos(2*np.pi*times*freq),np.ones(times.size)])
        rss.append(np.linalg.lstsq(A.T,mags)[1][0])
    return np.array(rss)

## given a set of times, determine frequencies to test for l-s
## find minimum, maximum, and fineness of grid
def get_freqs(times):
    ## look for periods no shorter than 1 day
    period_min = 1.0
    freq_max = 1. / period_min
    ## look for periods no longer than min(500days,length of oberervation time)
    ## period_max = min((times[-1] - times[0]),100.)
    period_max = 100.
    freq_min =  1. / period_max
    freq_del = .01 * freq_min
    return np.linspace(freq_min,freq_max,(freq_max - freq_min)/freq_del)


## alternative method for getting frequencies
## linspace in period
def get_freqs2(times):
    ## look for periods no shorter than 1 day
    period_min = 50.0
    period_max = 1000.0
    delta = .1
    periods = np.linspace(period_min,period_max,(period_max - period_min)/.1)
    return 1./periods



## what is a good data format here
## list of tfes
def lomb_multiband(tfes,freqs):
    rsses = np.empty(freqs.shape[0]*len(tfes)).reshape((freqs.shape[0],len(tfes)))
    for ii in range(len(tfes)):
        tfe = tfes[ii]
        times = tfe[:,0]
        mags = tfe[:,1]
        mags_errors = tfe[:,2]
        rsses[:,ii] = lomb(times,mags,mags_errors,freqs)
    return rsses
        
## create fake sinusoidal data, examine
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    A = 2
    p = .04
    w = .619
    b0 = 3
    n = 101
    times = np.linspace(0,6,n)
    mags_errors = 1.*np.ones(n)
    mags = b0 + A*np.sin(2*np.pi*times*w + p) + np.random.normal(0,mags_errors[0],n)
    ## plot fake sinusoidal data
    plt.plot(times,mags,'ro')
    plt.show()
    ## get freqencies
    freqs = get_freqs(times)
    ## find period for fake sinusoidal data
    rss = lomb(times,mags,mags_errors,freqs)
    print rss
    plt.plot(freqs,rss)
    plt.axvline(x=w,linewidth=3,color='r')
    plt.show()

