###
### tools for visualizing curves
###
### by James Long
### date August 8, 2011
###

import numpy as np
from matplotlib import pyplot as plt

###
### write code to save to file
###

## shows folded and/or unfolded curves
def plot_curve(tfe,true,naive,hrl,model):
    q1 = tfe[:,3] == 1
    q0 = tfe[:,3] == 0
    plt.plot(tfe[q1,0],tfe[q1,1],'r.')
    plt.plot(tfe[q0,0],np.repeat(tfe[:,1].min(),q0.sum()),'b.')
    plt.xlabel("Time")
    plt.ylabel("Flux")
    if true:
        ptimes = np.linspace(tfe[:,0].min(),tfe[:,0].max(),1000)
        plt.plot(ptimes,model.true.mean(ptimes),'k-')
    if naive:
        ptimes = np.linspace(tfe[:,0].min(),tfe[:,0].max(),1000)
        plt.plot(ptimes,model.naive.mean(ptimes),'b-')
    if hrl:
        ptimes = np.linspace(tfe[:,0].min(),tfe[:,0].max(),1000)
        plt.plot(ptimes,model.hrl.mean(ptimes),'g-')
    plt.show()
