

path = '/Users/jwrichar/Documents/CDI/CensoredData/'

miras = read.table(paste(path,"data/new_periods.dat",sep=""))

sum( abs((miras[,2] * 2 - miras[,4]) / miras[,4]) < 0.05) / dim(miras)[1]
# 0.1264775 of Miras are 'double' the LS period
sum( abs((miras[,2] * 3 - miras[,4]) / miras[,4]) < 0.05) / dim(miras)[1]
# 0.005122143 are triple

pdf(paste(path,"plots/mira_period_amplitude_relation.pdf",sep=""),height=6,width=10)
par(mfrow=c(1,2),mar=c(5,5,2,2))
plot(miras[,2],miras[,3],pch=19,col="#00000015",log='x',xlim=c(50,1000),ylim=c(0.5,7),xlab="LS Period", ylab="LS Amplitude",cex=0.75)
plot(miras[,4],miras[,5],pch=19,col="#00000015",log='x',xlim=c(50,1000),ylim=c(0.5,7),xlab="Censored LS Period", ylab="Censored LS Amplitude",cex=0.75)
dev.off()

rho1 = cor(miras[,2],miras[,3]) # 0.01
rho2 = cor(miras[,4],miras[,5]) #  0.07

pdf(paste(path,"plots/mira_periods_LS_vs_Censored.pdf",sep=""),height=8,width=8)
par(mfrow=c(1,1),mar=c(6,6,1,1))
plot(miras[,2],miras[,4],pch=19,xlab="Lomb-Scargle Period",ylab="Censored LS Period",col="#00000050",cex.lab=1.5)
abline(0,0.5,col=2,lty=2,lwd=2); text(800,400,"half",pos=NULL,cex=1.5)
abline(0,1,col=2,lty=2,lwd=2); text(800,800,"same",pos=NULL,cex=1.5)
abline(0,2,col=2,lty=2,lwd=2); text(400,800,"double",pos=NULL,cex=1.5)
abline(0,3,col=2,lty=2,lwd=2); text(300,900,"triple",pos=NULL,cex=1.5)
dev.off()

double = which(abs((miras[,2] * 2 - miras[,4]) / miras[,4]) < 0.05)
triple = which(abs((miras[,2] * 3 - miras[,4]) / miras[,4]) < 0.05)

