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



## shows folded and/or unfolded curves
def plot_two_curves(tfe,censored_model,naive_model,
                    plot_name="plot.pdf",in_title=""):
    q1 = tfe[:,1] < 29
    q0 = tfe[:,1] > 29
    plt.figure(1)

    plt.subplots_adjust(hspace=.7)
    plt.subplot(311)
    plt.title("Unfolded " + in_title)
    plt.plot(tfe[q1,0],-tfe[q1,1],'r.')
    plt.plot(tfe[q0,0],-np.repeat(tfe[q1,1].max(),q0.sum()),'b.')
    plt.xlabel("Time")
    plt.ylabel("Flux")

    plt.subplot(312)
    plt.title("Not using Censored. Period: " + repr(naive_model))
    plt.plot((tfe[q1,0] % naive_model) / naive_model,
             -tfe[q1,1],'r.')
    plt.plot((tfe[q0,0] % naive_model) / naive_model,
             -np.repeat(tfe[q1,1].max(),q0.sum()),'b.')
    plt.xlabel("Phase")
    plt.ylabel("Flux")

    plt.subplot(313)
    plt.title(" Using Censored. Period: " + repr(censored_model))
    plt.plot((tfe[q1,0] % censored_model) / censored_model,
             -tfe[q1,1],'r.')
    plt.plot((tfe[q0,0] % censored_model) / censored_model,
             -np.repeat(tfe[q1,1].max(),q0.sum()),'b.')
    values = np.linspace(0,1,100)
    plt.xlabel("Phase")
    plt.ylabel("Flux")
    plt.savefig(plot_name, format='pdf')
    ##plt.show()
