"""
This file is part of the Censored Data Project
Copyright 2013, Joseph W. Richards, David W. Hogg, James Long

This code computes the density of the reported standard errors (s).

Issues:

"""

import numpy as np
import urllib2 as ulib
import time
from matplotlib import pyplot as plt
import pylab as P

#global path
path = '../'

#global catalog
cat_data = np.loadtxt(path + 'data/asas_class_catalog_v2_3.dat',\
                     usecols=(0,1,4,5,6,9,37), skiprows=1,\
                     dtype=[('ID','S20'), ('dID','i4'), ('class','S20'), ('Pclass',np.float), \
                            ('anom',np.float64), ('Pmira',np.float), ('P','f16')], delimiter=',')



def load_lc_from_web(ID):
    urlpath = 'http://www.astrouw.edu.pl/cgi-asas/asas_cgi_get_data?'
    ## ur looks like http://www.astrouw.edu.pl/cgi-asas/asas_cgi_get_data?000006+2553.2,asas3
    ur = ulib.urlopen( urlpath + ID + ',asas3')#open url
    miradata = np.loadtxt(ur, usecols=(0,2,7,11), \
                          dtype=[('t',np.float64), ('m',np.float64), ('e',np.float64),\
                                 ('grade','S2')])
    use = np.where(np.logical_and(np.logical_and(miradata['grade'] != 'D',miradata['grade'] != 'F'),\
                                  miradata['m'] != 99.999))[0]
    return miradata[use,:]




## get all light curves from ogle
p_mira = 0.75
miras = np.where(np.logical_and(cat_data['Pmira'] > p_mira , cat_data['anom'] < 3.))[0] 



errors = list()
for mira in miras:
    data = load_lc_from_web(cat_data[mira][0])
    np.savetxt("../data/mira_asas/" + cat_data[mira][0] + ".dat", data, delimiter = ',',fmt = '%s %s %s %s')
    indo = np.where(data['m'] != 29.999)
    eobs = data['e'][indo]
    errors.append(eobs)
    time.sleep(2)


er = np.empty(sum(map(len,errors)))
total = 0
for error in errors:
    er[total:(total+error.size)] = error
    total = total + error.size


P.hist(er,30)
P.xlabel('Reported Standard Error')
P.savefig("../plots/error_hist.pdf")
P.close()

P.hist(er,100)
P.xlabel('Reported Standard Error')
P.savefig("../plots/error_hist_100bins.pdf")
P.close()

P.hist(er[(er > .01)*(er < .10)],30)
P.xlabel('Reported Standard Error')
P.savefig("../plots/error_hist_narrow.pdf")
P.close()

