import numpy as np
import likelihood
from matplotlib import pyplot as plt
from glob import glob
import lomb

reload(likelihood)

## times2pred and pred2mu look good
t = np.linspace(0,10,1000)
sins, coss = likelihood.times2pred(t,w=2)
A = 4
B = 3
plt.plot(t,likelihood.pred2mu(sins,coss,A,B))
plt.show()




## if nothing is censored can't estimate mu_b or v_b
## if 1 obs is censored, can  make log-liklihood infinite
## by pu


t = np.linspace(0,10,1000)
sins, coss = likelihood.times2pred(t,w=.7)
A = 1.5
B = 1.3
C = 7.
mu = likelihood.pred2mu(sins,coss,A,B) + C
f = likelihood.mag2flux(mu)
mu2 = likelihood.flux2mag(f)

fig = plt.figure()
ax = fig.add_subplot(311)
ax.plot(t,mu)


ax2 = fig.add_subplot(312)
ax2.plot(t,f)

mu = likelihood.flux2mag(f)
ax3 = fig.add_subplot(313)
ax3.plot(t,mu2)
fig.show()



## add gaussian error to flux
mira_periods = "../data/mira_periods.dat"

miras = glob("../data/mira_asas/*")
mira = miras[3]
star = np.loadtxt(mira,
                  usecols=(0,1,2),skiprows=0)
ctimes = star[star[:,1] > 29.5,0]
times = star[star[:,1] < 29.5,0]
mags =  star[star[:,1] < 29.5,1]
errors = star[star[:,1] < 29.5,2]

## get lomb-scargle period
freqs = lomb.get_freqs2(times)
rss = lomb.lomb(times,mags,errors,freqs)
period = 1. / freqs[np.argmin(rss)]
## from lomb, get A,B,C values


reload(likelihood)
sin_c, cos_c = likelihood.times2pred(ctimes,1./period)
sin_uc, cos_uc = likelihood.times2pred(times,1./period)
f = likelihood.mag2flux(mags)
e = likelihood.magerr2fluxerr(mags,errors)
D = np.array([ np.sin(2*np.pi*times/period), 
               np.cos(2*np.pi*times/period),
               np.ones(times.size)])
A,B,C = np.linalg.lstsq(D.T,mags)[0]

## plot this model
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(times,-mags,'o')
times_pred = np.linspace(np.min(times),np.max(times),1000)
mags_pred = A*np.sin(2.*np.pi*times_pred / period) + B*np.cos(2.*np.pi*times_pred / period) + C
ax.plot(times_pred,-mags_pred)
ax.plot(ctimes,np.min(-mags)*np.ones(ctimes.size),'o')
fig.show()


mu_b = np.min(f)
v_b = 1.0

## TODO:
## get rid of censored data and try to optimize over A,B,C
## == set mu_b and v_b very low
## 1. starting from A,B,C obtained by lomb
## 2. from nearby A,B,C
## 3. from far away A,B,C
## how many iterations, how often convergence (should always converge)
## == change mu_b and v_b, do directions affect likelihood in expected manner
## == let mu_b and v_b vary but have minimums for them (mu_b = 0 min?)
## should get same solutions but mu_b and v_b should go towards minimums


ll = likelihood.log_likelihood_fixed_w(sin_c,cos_c,
                                       sin_uc,cos_uc,
                                       f,e,A,B,C,mu_b,v_b)
ll

ll = likelihood.log_likelihood_fixed_w(sin_c,cos_c,
                                       sin_uc,cos_uc,
                                       f,e,A+.1,B,C,mu_b,v_b)
ll


ll = likelihood.log_likelihood_fixed_w(sin_c,cos_c,
                                       sin_uc,cos_uc,
                                       f,e,A+.5,B,C,mu_b,v_b)
ll


ll = likelihood.log_likelihood_fixed_w(sin_c,cos_c,
                                       sin_uc,cos_uc,
                                       f,e,A+1.,B,C,mu_b,v_b)
ll






ctimes
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(times,mags,'o')
fig.show()


