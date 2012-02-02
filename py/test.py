import numpy as np
import scipy.optimize
import scipy.stats



x_rands = scipy.stats.norm().rvs(100)
y_rands = 2.7*x_rands + 10 + scipy.stats.norm().rvs(100)



# def squared_error(coeffs):
#     return(sum(np.power(y_rands - x_rands*coeffs[1] - coeffs[0],2)))

def squared_error(coeffs):
    return(sum(np.power(y_rands - coeffs[0],2)))



coeffs = [np.mean(y_rands),0]
print coeffs
coeffs = scipy.optimize.fmin(squared_error,coeffs)
print coeffs
print np.mean(y_rands)
