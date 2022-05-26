#' @title Estimate Log-likelihood for Gaussian random fields when Betas are given
#' @import data.table
#' @useDynLib gpuLik
#' @export



       Prof2dBetas <- function(Betas, #a p x 1 R matrix  given by the user 
                               prof2list,
                               Nobs,  # number of observations.
                               Nparam,
                               Ndata,
                               detVar, # vclVector
                               ssqY,   # vclMatrix
                               XVYXVX,   # vclMatrix
                               jacobian, # vclVector  #form = c("loglik", "profileforBeta"),
                               cilevel){
  
   # prof2list = list(Betas1 <- Betas1,
   #                  Betas2 <- Betas2)
   # 
   # Betas = do.call(expand.grid, prof2list)  
  
  if(!is.matrix(Betas))
  Betas<-as.matrix(Betas)
  
  #boxcox = c(1, 0, setdiff(boxcox, c(0,1)))
  Ncov = ncol(Betas)
  m = nrow(Betas)
  #Nparam = nrow(params)
  
  
  
  detVar <- matrix(rep(detVar, Ndata), nrow=Nparam)
  jacobian <- do.call(rbind, replicate(Nparam, jacobian, simplify = FALSE))
  XTVinvX <- XVYXVX[ , (ncol(XVYXVX)-Ncov+1):ncol(XVYXVX)]
  bTDinva <- as.matrix(XVYXVX[ , 1:Ndata], nrow=Ncov, Ncol=Ndata)
  midItem <- matrix(88, nrow=Nparam, ncol=Ndata)
  ssqBeta0 <- rep(0, length=Nparam)   # dosen't depend on lambda
  ssqForBetas <- matrix(0, nrow=m, ncol=1)
  likForBetas<- matrix(0, nrow=m, ncol=1)
  
  
  # one = ssqY - 2*beta^T * bTDinva + ssqBeta
  # n*log(one/n)+ logD +jacobian + n*log(2*pi) + n
  # takes about over 7 minutes for 126,000 params
  for(beta in 1:m){  
    
    for (i in 1:Nparam){
      # for each of the beta, calculate the beta^T * bTDinva, 1 x Ndata
      midItem[i,] <- Betas[beta, ] %*% as.matrix(bTDinva[((i-1)*Ncov+1) : (i*Ncov), ], nrow=Ncov)
      
      # ssqBeta = beta^T * (b^T D^(-1) b) * beta
      mat <- XTVinvX[((i-1)*Ncov+1) : (i*Ncov), ]
      mat[upper.tri(mat)] <- t(mat)[upper.tri(mat)]
      ssqBeta0[i] <- Betas[beta, ] %*% mat %*% (Betas[beta, ])
    }
    
    ssqBeta <- do.call(cbind, replicate(Ndata, ssqBeta0, simplify=FALSE))   # ssq same for different data sets
    one <- ssqY - 2*midItem + ssqBeta
    
    temp <- Nobs*log(one/Nobs) + detVar + Nobs + Nobs*log(2*pi) + jacobian
    likForBetas[beta,] = min(temp)
    ssqForBetas[beta,] <- ssqBeta0[which(temp == min(temp, na.rm = TRUE), arr.ind = TRUE)[1]]
  }
  
  LogLikBetas = -0.5*likForBetas
  result = data.frame(cbind(Betas,LogLikBetas))
  colnames(result) <- c('Betas1','Betas2',"LogLik")
  
  maximum <- max(LogLikBetas)
  breaks = maximum - qchisq(cilevel,  df = 2)/2
 
  lMatrix = matrix(result[,'LogLik'], length(prof2list[[1]]), length(prof2list[[2]]))
  # par(cex.lab=1.2)
  # par(mfrow = c(1, 1))
  contour(prof2list[[1]], prof2list[[2]], lMatrix,
          col = par("fg"), lty = par("lty"), lwd = par("lwd"), levels = c(breaks+1, breaks+2, breaks-3, breaks-2, breaks-1, breaks), 
          add = FALSE, xlab = "Beta1", ylab = "Beta2")
  
  # filled.contour(prof2list[[1]], prof2list[[2]], lMatrix, levels = c(breaks-4, breaks-3, 
  #                                                                    breaks-2, breaks-1, breaks),
  #                color = function(n) hcl.colors(n, "ag_Sunset"),
  #                plot.title={
  #                  title(xlab = "Intercept", ylab = "Elevation",main = "contour plot")
  #                  #abline(h=trueParam['nugget'], col='red')
  #                  #abline(v=trueParam['range']/100, col = "red")
  #                  })
  

  # Sprob = c(0,0.2, 0.5, 0.8, 0.9, 0.95, 0.999)
  # likCol = mapmisc::colourScale(drop(result$LogLik), breaks=max(lMatrix, na.rm=TRUE)- qchisq(Sprob, df=2), style='fixed', col='Spectral')
  # points(x = result[,1], y = result[,2], pch=20, col=likCol$plot, cex=0.8)
  # points(result[which.max(result$LogLik),],pch=20)
  # mapmisc::legendBreaks('bottomright', breaks = rev(Sprob), col=likCol$col)


  # cov1 <- apply(lMatrix, 2, max)
  # plot(prof2list$Betas2,cov1 -breaks)
  # f1 <- approxfun(prof2list$Betas2,cov1 -breaks)
  # curve(f1(x), add = TRUE, col = 2, n = 1001)
  # abline (h=0)
  # ci <- rootSolve::uniroot.all(f1, lower = -1, upper = 3)
  # abline(v=c(ci), lty=2)
  # intercept <- apply(lMatrix, 1, max)
  # plot(prof2list$Betas1,intercept-breaks)
  # 
  # profileLogLik <- as.data.frame(cbind(prof2list$Betas1,intercept-breaks))
  # colnames(profileLogLik) <- c("x1",'profile')
  # datC2 = geometry::convhulln(profileLogLik)
  # allPoints = unique(as.vector(datC2))
  # toTest = profileLogLik[allPoints,]
  # toTest[,'profile'] = toTest[,'profile'] + 0.1
  # inHull = geometry::inhulln(datC2, as.matrix(toTest))
  # toUse = profileLogLik[allPoints,][!inHull,]
  # toTest = profileLogLik[allPoints,]
  # 
  # plot(profileLogLik$x1, profileLogLik$profile, cex=.2, xlab="Betas1", ylab="profileLogL")
  # points(toTest, col='red', cex=0.6)
  # points(toUse, col='blue', cex=0.6, pch=3)
  # abline(h =0, lty = 2, col='red')
  # 
  # interp1 = mgcv::gam(profile ~ s(x1, k=nrow(toUse),  m=1, fx=TRUE), data=toUse)
  # prof = data.frame(x1=seq(min(toUse$x1)-0.2, max(toUse$x1)+0.2, len=101))
  # prof$z = predict(interp1, prof)
  # 
  # lines(prof$x1, prof$z, col = 'green')
  # lower = min(profileLogLik$x1)
  # upper = max(profileLogLik$x1)
  # f1 <- approxfun(prof$x1, prof$z)
  # MLE <- optimize(f1, c(lower, upper), maximum = TRUE, tol = 0.0001)$maximum
  # ci<-rootSolve::uniroot.all(f1, lower = lower, upper = upper)
  # abline(v =c(MLE,ci), lty = 2, col='red')
  
  #plot(profileLogLik_intercept$intercept,profileLogLik_intercept$profile)

  Theoutput <- list(dataforplot=result,
                    breaks =breaks,
                    ssqForBetas = ssqForBetas)
  
  
  Theoutput
  
  
}
  
  
  





 # resultbeta1 = data.table::as.data.table(cbind(-0.5*LogLikBetas, Betas[,2]))
 # colnames(resultbeta1) <- c("LogLik", "Beta1")
 # profileLogLik_beta1 <- resultbeta1[, .(profile=max(.SD)), by=Beta1]
 # 
 # 
 # resultintercept = data.table::as.data.table(cbind(-0.5*LogLikBetas, Betas[,1]))
 # colnames(resultintercept) <- c("LogLik", "intercept")
 # profileLogLik_intercept <- resultintercept[, .(profile=max(.SD)), by=intercept]
 














  
  