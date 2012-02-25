import numpy as np
from scipy.optimize import fmin
from scipy import special


## this is weighted least squares, weights on points are given uncertainties
## we ignore censored observations here
def naive_opt(myData):
    ## compute weight matrix, remember errors are in SD, not var
    Wsqrt = np.diag(myData.known_errors[:,0])
    ## multiply predictors and response by weight matrix
    X = np.dot(Wsqrt,myData.known_predictors)
    Y = np.dot(Wsqrt,myData.known_response)
    ## now compute the OLS estimate
    XtXinv = np.linalg.inv(np.dot(X.T,X))
    XtY = np.dot(X.T,Y)
    return np.dot(XtXinv,XtY)


## a simple version of our model where:
## 1) sigmas = s's (i.e. reported standard deviations are correct)
## 2) sigma_mu = 0 (no model uncertainty)
## 3) known censoring threshold
## compute_censored=False ignores censoring, use this to test
## that optimization methods are working
def simple_model_likelihood(params,myData,b,compute_censored):
    params = np.array(params).reshape((len(params),1))
    ## compute (-2.0 * log likelihood) for uncensored observations
    normalized_differences = (myData.known_response - np.dot(myData.known_predictors,params)) / myData.known_errors
    lik_uncensored = np.sum(np.square(normalized_differences))
    ## compute (-2.0 * log likelihood) for censored observations
    lik_censored = 0.0
    if compute_censored:
        ## NOTE: np.mean known errors is a total hack, this assumes errors are same
        normalized_differences = (b - np.dot(myData.censored_predictors,params)) / np.mean(myData.known_errors)
        lik_censored = -2.0 * np.sum(np.log(special.erf(normalized_differences/np.sqrt(2.0)) + 1.0))
    ## return sum of liklihoods
    return lik_uncensored + lik_censored


class SimParameters:
    def __init__(self,p=1,n=10):
        self.beta = np.random.normal(size=p+1).reshape((p+1,1))
        self.sigmas = np.ones(n).reshape((n,1))
        self.n = n

class SimData:
    def __init__(self,Params):
        p = Params.beta.shape[0] - 1
        self.ss = Params.sigmas ## because for now errors are true
        self.X = (np.random.uniform(low=-1.0,high=1.0,size=Params.n)).reshape((Params.n,p))
        self.X = np.insert(self.X,0,np.ones(Params.n),1) ## make the first column all ones
        self.epsilon = Params.sigmas*np.random.normal(size=Params.n).reshape((Params.n,1))
        self.Y = np.dot(self.X,Params.beta) + self.epsilon
    ## truncate the data at some level, put into Data object
    def truncate(self,b=0.0):
        to_truncate = (self.Y[:,0] < b)
        known_predictors = self.X[np.invert(to_truncate),:]
        known_response = self.Y[np.invert(to_truncate),:]
        known_errors = self.ss[np.invert(to_truncate),:]
        censored_predictors = self.X[to_truncate,:]
        self.myData = Data(known_predictors,known_response,known_errors,censored_predictors)

## dat object contains 4 items, this should be pretty stable
class Data:
    def __init__(self,known_predictors,known_response,known_errors,censored_predictors):
        self.known_predictors = known_predictors
        self.known_response = known_response
        self.known_errors = known_errors
        self.censored_predictors = censored_predictors
    def pprint(self):
        print "known predictors:"
        print self.known_predictors
        print "known response:"
        print self.known_response
        print "known errors:"
        print self.known_errors
        print "censored_predictors:"
        print self.censored_predictors

    

if __name__ == "__main__":
    ## generate parameters and data for the model
    p = 1
    n = 100
    params = SimParameters(p,n)
    mySimData = SimData(params)
    ## censor measurements at b
    b = 0.0
    mySimData.truncate(b)
    ## run several optimization routines
    naive_min = naive_opt(mySimData.myData)
    v0 = (p+1)*[0.0]
    v = fmin(simple_model_likelihood,v0,args=(mySimData.myData,b,False),maxiter=10000,maxfun=10000)

    print "naive, ls solution"
    print naive_min
    print "naive solution using fmin to optimize -> should be same as ls"
    print v

    v0 = (p+1)*[0.0]
    v = fmin(simple_model_likelihood,v0,args=(mySimData.myData,b,True),maxiter=10000,maxfun=10000)
    print "simple model that includes truncation"
    print v
    print "truth"
    print params.beta
    



# what are the parameters I am optimizing
# 1) coefficients in linear model, 1 / column
# 2) 
##def compute_likelihood(known_predictors,known_response,known_uncertainty,censored_predictors):
    

## this is the data for a l.c. after selecting period, harmonics, ect



## data meta_information
## 1) p = nrow(known_predictors) --> 1 param per row that we have to optimize over
## 2) n1 = ncol(known_predictors) --> num. non-censored obs
## 3) n2 = ncol(censored_predictors) --> num censored predictors


## parameters
## 1) p params for coefficients in linear model
## 2) model uncertainty param
## 3) params for linking true uncertainty with si (want to make optimizing over this optional, flexible)
## 4) priors on the sigma_i
## 5) truncation uncertainty

#### first approximation
## 1. assume no model uncertainty, fixed truncation threshold, known sigmas, 1-d problem
##      code this up (essentially optimizing over a single beta


### think about what the default model is        




## TODO: why does this happen with python?
# >>> b
# array([[-1.10553941, -1.46065012],
#        [ 0.29458701,  0.14580707],
#        [ 0.06907306,  0.77544002],
#        [ 0.70967028, -1.48951893],
#        [-0.21702106,  0.28071874]])
# >>> b[[True,True,False,False,False]]
# array([[ 0.29458701,  0.14580707],
#        [ 0.29458701,  0.14580707],
#        [-1.10553941, -1.46065012],
#        [-1.10553941, -1.46065012],
#        [-1.10553941, -1.46065012]])
# >>> b[np.array([True,True,False,False,False])]
# array([[-1.10553941, -1.46065012],
#        [ 0.29458701,  0.14580707]])


