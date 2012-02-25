##################
################## OBSOLETE CODE ---> WORKING CODE NOW IN censoring.py
##################



########## 
########## 
########## CENSORED REGRESSION ALGORITHM 
########## 
##########
########## by James Long 
########## date: 11/8/2011 
########## 

import cProfile
import numpy as np
import scipy.optimize
import scipy.stats
import visualize
import pstats
from scipy import special

def gaussian_pdf_1d(x,mean,var):
    return (1./np.sqrt(2.* np.pi * var))*np.exp(-(0.5*(x-mean)**2)/var)

def gaussian_cdf_1d(x,mean,var):
    return .5*(1. + special.erf((x-mean)/np.sqrt(2.*var)))

# physical explanation, data, 
class LightCurve:
    def __init__(self, data, model, prior):
        self.data = data
        self.model = model
        self.prior = prior
    ## makes a plot of the light curv
    def plot(self,true=False,naive=False,hrl=False):
        visualize.plot_curve(self.data.get_all(),true=true,naive=naive,hrl=hrl,model=self.model)
    def naive_optimize(self):
        tfe = self.data.get_all()
        tfe = tfe[tfe[:,3]==1,:]
        def cost(p, const):
            self.model.naive.set_opt_params(p)
            return ((self.model.naive.mean(tfe[:,0]) - tfe[:,1]) / tfe[:,2])**2
        p = self.model.naive.get_opt_params()
        const = self.model.naive.get_const_params()
        betterp = scipy.optimize.leastsq(cost, p, const)
    def ln_likelihood(self):
        self.bs = self.prior.bsample(100)
        self.sigmas = self.prior.sigmasample(100)
        return sum(map(lambda datum: self.ln_likelihood_point(datum,self.bs,self.sigmas), self.data))
    #compute p(D_n|theta) 
    def ln_likelihood_point(self, datum, bs, sigmas):
        return np.log(np.mean(self.likelihood_point(datum,self.bs,self.sigmas)))
    def likelihood_point(self, datum, bs, sigmas):
        t, q, f, s = datum
        mu = self.model.hrl.mean(t)
        variances = self.model.hrl.get_modelvariance() + sigmas**2
        if q == 1:
            ## p(qi|b,sigma,theta,I) * p(f|q,sigma,theta,I) * p(s|q,sigma,theta,I)   where qi = 1
            return ((1 - gaussian_cdf_1d(self.bs,mu,variances))*
                    gaussian_pdf_1d(f,mu,variances)*
                    gaussian_pdf_1d(self.sigmas,s,self.prior.get_vsig2()))
            ## p(qi|b,sigma,theta,I) where qi = 0 (don't need other 2 terms b/c they = 1 here)
        return gaussian_cdf_1d(bs,mu,variances)
    def optimize(self):
        def cost(p):
            print self.model.hrl.get_params_names()
            print self.model.hrl.get_params()
            self.prior.set_params(p[0],p[1],p[2],p[3],p[4])
            self.model.hrl.set_opt_params(p[4:])
            return -2. * self.ln_likelihood()
        ## optimize over p params
        p = list(self.prior.get_params())
        p.extend(list(self.model.hrl.get_opt_params()))
        p = np.array(p)
        ## do not optimize over constant params (e.g. omega)
        const = self.model.hrl.get_const_params()
        ## run the optimizer over p, using function cost, keeping const parameters
        betterp = scipy.optimize.fmin(cost, p)


class CatalogPrior:
    def __init__(self):
        pass
    ## params = B, VB, S, VS, vsig2
    def get_params_names(self):
        return ["B","VB","S","VS","vsig2"]
    def set_params(self,B,VB,S,VS,vsig2):
        self.B = B
        self.VB = VB
        self.S = S
        self.VS = VS
        self.vsig2 = vsig2
    def get_params(self):
        return self.B, self.VB, self.S, self.VS, self.vsig2
    ## by sampling the b's and sigmas's from here, we can easily change the priors
    ## we want to use, following nat's suggestion
    def bsample(self,num=1000):
        return np.random.normal(loc=self.B,scale=np.sqrt(self.VB),size=num)
    def sigmasample(self,num=1000):
        return np.random.normal(loc=self.S,scale=np.sqrt(self.VS),size=num)
    def get_vsig2(self):
        return self.vsig2


class Data:
    def __init__(self, times, qs, fs, ss):
        self.times = times
        self.qs = qs
        self.fs = fs
        self.ss = ss
    def __getitem__(self, k):
        return self.times[k], self.qs[k], self.fs[k], self.ss[k]
    def get_all(self):
        return np.column_stack((self.times[:,np.newaxis],self.fs[:,np.newaxis],self.ss[:,np.newaxis],self.qs[:,np.newaxis]))

class PeriodicVariable:
    def __init__(self):
        pass
    def get_params_names(self):
        return ["omega","A","B","modelvariance","base"]
    ## choose which parameters to optimize
    def which_to_opt(self,to_opt):
        self.variables = to_opt
        self.constants = np.invert(to_opt)
    ## manipulate opt parameters
    def get_opt_params(self):
        return self.params[self.variables]
    def set_opt_params(self, opt):
        self.params[self.variables] = opt
    ## manipulate constant parameters
    def get_const_params(self):
        return self.params[self.constants]
    def set_const_params(self, const):
        self.params[self.constants] = const
    def set_params(self, params):
        self.params = np.array(params)
    ## do things to all parameters
    def get_params(self):
        return self.params
    def mean(self, time):
        return self.params[1] * np.cos(self.params[0] * time) + self.params[2] * np.sin(self.params[0] * time) + self.params[4]
    ## need direct access to modelvariance
    def get_modelvariance(self):
        return self.params[3]

## holds three model object (e.g. PeriodicVariable)
## object 1 has true parameter values
## object 2 contains parameter from naive
## object 3 contains parameter from our model
class VariableObject:
    def __init__(self,true,naive,hrl):
        self.true = true
        self.naive = naive
        self.hrl = hrl
    
### 
### generate fake like curve
###
def real_times(filename="OGLE-LMC-LPV-59224.dat"):
    # get real times and flux errors
    tfe = np.genfromtxt(filename)
    return(tfe[:,0])

## returns a practice sinusoid light curve
def fake_lc(times = np.arange(10000,dtype='float64')/100):
    ## create VariableObject myPeriodicVariable setup
    which_to_optimize = [False,False,False,False,True]
    myPeriodicVariable = PeriodicVariable()
    myPeriodicVariable.set_params([(1/5.5),.3,1,.1,6])
    myPeriodicVariable.which_to_opt(np.array(which_to_optimize))
    myPeriodicVariableNaive = PeriodicVariable()
    myPeriodicVariableNaive.which_to_opt(np.array(which_to_optimize))
    myPeriodicVariableHRL = PeriodicVariable()
    myPeriodicVariableHRL.which_to_opt(np.array(which_to_optimize))
    myVariableObject = VariableObject(myPeriodicVariable,myPeriodicVariableNaive,myPeriodicVariableHRL)
    ## myCatalogPrior setup
    myCatalogPrior = CatalogPrior()
    myCatalogPrior.set_params(0.2,.1,.1,.1,.1)
    ## myData setup
    sigmas = myCatalogPrior.sigmasample(num=times.size)
    errors = sigmas + scipy.stats.norm.rvs(loc=0,scale=np.sqrt(myCatalogPrior.get_params()[4]),size=sigmas.size)
    fluxes = myPeriodicVariable.mean(times) + sigmas
    bs = myCatalogPrior.bsample(num=times.size)
    ## drawing qi's according to equation 9 (qi is bernoulli)
    mus = myPeriodicVariable.mean(times)
    qs_prob = gaussian_cdf_1d(bs,mus,sigmas**2 + myPeriodicVariable.get_modelvariance())
    qs = 1*(scipy.stats.uniform.rvs(loc=0,scale=1,size=qs_prob.size) > qs_prob)
    ## load entries into data
    myData = Data(times,qs,fluxes,errors)
    ## return everything
    return([myVariableObject,myData,myCatalogPrior])

if __name__ == "__main__":
    myVariableObject, myData, myCatalogPrior = fake_lc()
    myLightCurve = LightCurve(myData,myVariableObject,myCatalogPrior)
    myLightCurve.model.naive.set_params([1/5.5,.3,1,.1,-10])
    myLightCurve.model.hrl.set_params([1/5.5,.3,1,.1,-10])
    myLightCurve.naive_optimize()
    print "::parameter names::"
    print myLightCurve.model.naive.get_params_names()
    print "::naively optimized parameters::"
    print myLightCurve.model.naive.get_params()
    print "::true parameters::"
    print myLightCurve.model.true.get_params()
    myLightCurve.plot(true=True,naive=True,hrl=False)
    myLightCurve.optimize()



## TODO:
## option so don't estimate period
## either don't update value or regularize
## subtract mean from lc
## force optimize to keep variances > 0

## diagnostics
## 1. have fmin optimize for just constant, this should be easy
## 2. examine if variances are ever negative
##     - probably want to build in feature that makes this impossible
##     - the sigmas appear squared a lot (sigmas**2, so their mean could be 0)
## 3. at some points normal variances are sum of two variances e.g.
##            variances = self.model.hrl.get_modelvariance() + sigmas**2
##    could be problem here where one is very large, and another is negative


## 1. moving the calls to gaussian pdf and cdf inside objects
##    would allow you to change priors easily -- right now
##    several thing call these functions
## 3. legend on plot
## 4. do better mapping

