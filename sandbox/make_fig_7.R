### 
### 
### MAKE FIGURE 7 IN PAPER 
###


periods <- read.table("../data/new_periods.dat")
periods[,1] <- as.character(periods[,1])
head(periods)


plot(periods[,2],periods[,4])

wrong <- c("000842-8611.3")
star <- periods[periods[,1]==wrong,]

points(star[2],star[4],col='red',pch=20,cex=2)


periods10 <- read.table("../data/new_periods_10.dat")
periods10[,1] <- as.character(periods10[,1])
points(periods10[,2],periods10[,4],col="blue")



dev.new()
plot(periods[,3],periods[,5],main="amps")
points(star[3],star[5],col='red',pch=20,cex=2)
abline(0,1,col='grey')

dev.new()
plot(periods[,2],periods[,3],main="period vs. amplitude",
     ylim=c(0,5),xlim=c(0,600))
points(star[2],star[3],col='red',pch=20,cex=2)



dev.new()
plot(periods[,4],periods[,5],main="period vs. amplitude NEW",
     ylim=c(0,5),xlim=c(0,600))
points(star[2],star[3],col='red',pch=20,cex=2)
