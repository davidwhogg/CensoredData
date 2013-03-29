
## simulated mocked up light curves w/non-detections from real, well-observed LCs

##########################################################################################
## functions

# magnitude to flux conversion
mag2flux = function(mag,magerr){
  flux.zp = 3.67*10^(-9) # V-band flux zero-point
  mag.zp = 2.5*log10(flux.zp)
  flux = 10^((mag.zp-mag)/2.5)
  fluxerr = sqrt(magerr^2 * (flux*log(10)*(1/2.5))^2)
  return(list(flux=flux,fluxerr=fluxerr))
}

# flux to mag conversion
flux2mag = function(flux,fluxerr){
  flux.zp = 3.67*10^(-9) # V-band flux zero-point
  mag.zp = 2.5*log10(flux.zp)
  mag = -2.5 * log10(flux) + mag.zp
  magerr = sqrt(fluxerr^2 * (-2.5/(flux*log(10)))^2)
  return(list(mag=mag,magerr=magerr))
}


################# create simulated LC from existing LC

makeSim = function(time,flux,fluxerr,p,sigmodel,eps=1e-16){
  n=length(flux)
  muflux = flux*p
  pred.err = predict(sigmodel,newdata=data.frame(flux=log10(muflux)),se.fit=TRUE)
  mufluxerr = pred.err$fit
  sigfluxerr = pred.err$se.fit

  fluxerrnew = 10^rnorm(n,mean=mufluxerr,sd=sigfluxerr)+eps
  fluxnew = rnorm(n,mean=flux*p,sd=fluxerrnew)

  nondetect = ifelse(fluxnew - 5*fluxerrnew <=0,1,0)
  
  return(list(flux=fluxnew,fluxerr=fluxerrnew,nondetect=nondetect))
}

#####################################################################################

#####################################################################################
################# miras
miras = system("ls /Users/jwrichar/Documents/CDI/LS_upperlim/miras/",intern=TRUE)
for(ii in 1:length(miras)){
  data = read.table(paste("/Users/jwrichar/Documents/CDI/LS_upperlim/miras/",miras[ii],sep=""))
#  cat(miras[ii]," # non-detect",sum(data[,2]==29.999)," # detect ",sum(data[,2]!=29.999),"\n")
}
path = '/Users/jwrichar/Documents/CDI/CensoredData/'
# use 235627-4947.2: no non-detections over 408 epochs
mira.data = read.table(paste(path,"data/lc_235627-4947.2.dat",sep=""))

use = mira.data[,2]<29.999
P = 266.6286 # period
# plot:
plot((mira.data[use,1]/P) %% 1,mira.data[use,2],pch=19,ylim=c(max(mira.data[use,2]),min(mira.data[,2])),xlim=c(0,1),xlab="Phase",ylab="V mag",cex=0.5)
rug((mira.data[mira.data[,2]==29.999,1]/P) %% 1,col=4,ticksize=0.025,lwd=1)
arrows((mira.data[use,1]/P) %% 1,mira.data[use,2]+mira.data[use,3],(mira.data[use,1]/P) %% 1,mira.data[use,2]-mira.data[use,3],length=0)


## get empirical flux - flux-error relationship
mags = magerrs = NULL
for(ii in 1:length(miras[1:100])){
  data = read.table(paste("/Users/jwrichar/Documents/CDI/LS_upperlim/miras/",miras[ii],sep=""))
  mags = c(mags,data[data[,2]!=29.999,2])
  magerrs = c(magerrs,data[data[,2]!=29.999,3])
}

fluxes = mag2flux(mags,magerrs)

#plot(mags,magerrs)

plot(fluxes$flux,fluxes$fluxerr,log='xy')

# empirical flux - fluxerror relationship
lm.fit = lm(fluxerr~flux,data=lapply(fluxes,log10))
pred.model = predict(lm.fit,newdata=data.frame(flux=(seq(-15,-11,.01))),se.fit=TRUE,interval="prediction")
lines(10^(seq(-15,-11,.01)),10^pred.model$fit[,1]+1e-16,col=4,lwd=2)
lines(10^(seq(-15,-11,.01)),10^pred.model$fit[,2],col=3,lwd=2,lty=2)
lines(10^(seq(-15,-11,.01)),10^pred.model$fit[,3],col=3,lwd=2,lty=2)


mira.data.f = mag2flux(mira.data[,2],mira.data[,3])
mira.data.flux = mira.data
mira.data.flux[,2] = mira.data.f$flux
mira.data.flux[,3] = mira.data.f$fluxerr

                                        # plot:
## plot((mira.data.flux[use,1]/P) %% 1,mira.data.flux[use,2],pch=19,xlim=c(0,1),xlab="Phase",ylab="V-band flux",cex=0.5)
## rug((mira.data.flux[mira.data.flux[,2]==29.999,1]/P) %% 1,col=4,ticksize=0.025,lwd=1)
## arrows((mira.data.flux[use,1]/P) %% 1,mira.data.flux[use,2]+mira.data.flux[use,3],(mira.data.flux[use,1]/P) %% 1,mira.data.flux[use,2]-mira.data.flux[use,3],length=0)


#### simulate LCs
set.seed(1)
#pdf("/Users/jwrichar/Documents/CDI/LS_upperlim/plots/mira_simulated.pdf",height=10,width=10) 
#par(mfrow=c(2,2),mar=c(4.5,4.5,1,1))
pvec = c(1,0.5,0.25,0.1,0.05,0.02,0.01, 0.007, 0.005, 0.002)
median.mag = rep(NA, length(pvec))
for(ii in 1:length(pvec)){
  new.lc = makeSim(mira.data.flux[,1],mira.data.flux[,2],mira.data.flux[,3],p=pvec[ii],sigmodel=lm.fit,eps=1e-15)
  median.mag[ii] = flux2mag(median(new.lc$flux),0)$mag
  
  mira.new = cbind(mira.data[,1],flux2mag(new.lc$flux,new.lc$fluxerr)$mag,flux2mag(new.lc$flux,new.lc$fluxerr)$magerr)
  mira.new[new.lc$nondetect==1,2] = 29.999

  cat("Mira: ",pvec[ii], "Median mag =", median.mag[ii], " Number of detections = ", sum(new.lc$nondetect==0), "\n")
  ## use = mira.new[,2]<29.999
  ## plot((mira.new[use,1]/P) %% 1,mira.new[use,2],pch=19,xlim=c(0,1),xlab="Phase",ylab="V mag",cex=0.5,cex.axis=1.5,cex.lab=1.5,ylim=c(14.5,8))
  ## rug((mira.new[mira.new[,2]==29.999,1]/P) %% 1,col=4,ticksize=0.025,lwd=1)
  ## arrows((mira.new[use,1]/P) %% 1,mira.new[use,2]+mira.new[use,3],(mira.new[use,1]/P) %% 1,mira.new[use,2]-mira.new[use,3],length=0)
  ## legend("topleft",paste("flux/",round(1/pvec[ii]),sep=""),bty='n',pt.cex=0,cex=2)

 # write(t(mira.new),file=paste(path,"data/mira_sims/lc_235627-4947.2_dim_",pvec[ii],".dat",sep=""),ncolumns=3)
}
#dev.off()
