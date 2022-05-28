#' @title Estimate Log-likelihood for Gaussian random fields when Betas are given
#' @useDynLib gpuLik


     makeSymm <- function(m) {
         m[upper.tri(m)] <- t(m)[upper.tri(m)]
         return(m)
     }
  
#' @export

        Prof1dBetas <- function(Betas, #a m x p R matrix  given by the user 
                                cilevel,
                                Nobs,  # number of observations.
                                Ndata,
                                Nparam,
                                Ncov,
                                detVar, # 
                                detReml,
                                ssqY,   # 
                                XVYXVX,   # 
                                jacobian,
                                reml = FALSE,
                                I = NULL,
                                convexHull = FALSE){ 
  
  m <- nrow(Betas)
  if(m < 6){
    stop("need more values for accurate estimate")
  }
  

  detVar <- matrix(rep(detVar, Ndata), nrow=Nparam)
  jacobian <- do.call(rbind, replicate(Nparam, jacobian, simplify = FALSE))
  #dim(XVYXVX)
  Ucov <- Ncov-1
  XTVinvX <- XVYXVX[ , (ncol(XVYXVX)-Ncov+1):ncol(XVYXVX)]
  XVY <- matrix(XVYXVX[ , 1:Ndata], ncol=Ndata)
  
  #dim(XTVinvX)
  ## make each symmetric
  for (i in 1:Nparam){
    XTVinvX[((i-1)*Ncov+1) : (i*Ncov), ] <- makeSymm(XTVinvX[((i-1)*Ncov+1) : (i*Ncov), ])
  }
  
  if(!is.null(I)){
    Ncov = 1
  }

  LogLik_optimized = matrix(0, nrow=m, ncol=ncol(Betas))
  breaks <- rep(0, ncol(Betas))
  Table <- matrix(NA, nrow=ncol(Betas), ncol=4)
  colnames(Table) <-  c("MLE", paste(c('lower', 'upper'), cilevel*100, 'ci', sep = ''),"maximum")
  #index <- matrix(0, nrow=m, ncol=2)
  profBetas <- matrix(0, nrow=1001, ncol=2*ncol(Betas))
  

  for (a in 1:ncol(Betas)){
    if(!is.null(I)){
      a = I
      BetaSlice <- Betas
    }else{
      BetaSlice <- Betas[,a]
    }
  selectedrows <- (seq_len(Nparam)-1) * Ncov + a
  XTVinvX_deleted <- matrix(XTVinvX[-selectedrows,-a], ncol=Ucov)
  XTVinvX_a <- matrix(XTVinvX[-selectedrows, a],  ncol=1)
  XVY_deleted <- matrix(XVY[-selectedrows, ],ncol=Ndata)
  X_aVY <- matrix(XVY[selectedrows, ], ncol=Ndata)
  X_aVX_a <- XTVinvX[selectedrows, a]
  
  partA = matrix(0, nrow=Nparam, ncol=Ndata)
  partB = matrix(0, nrow=Nparam, ncol=Ndata)
  partC = matrix(0, nrow=Nparam, ncol=Ndata)
  partD = matrix(0, nrow=Nparam, ncol=Ndata)
  partE = matrix(0, nrow=Nparam, ncol=Ndata)
  
  if(Ndata == 1){
    if(Ucov==1){
      for (i in 1:Nparam){
        interval <- c(((i-1)*Ucov+1) : (i*Ucov))
        temp <- solve(XTVinvX_deleted[interval, ])
        #temp <- solve(XTVinvX_deleted[interval, ]) 
        #temp %*% XTVinvX_deleted[interval, ]
        # part (A) have 2 data sets
        partA[i,] = XVY_deleted[interval,] %*% temp %*% XVY_deleted[interval,]
        # part (B) have 2 data sets.   has beta
        partB[i,] = - XTVinvX_a[interval,] %*% temp %*% XVY_deleted[interval,]
        # part (C) no data sets.  has beta
        partC[i,] = XTVinvX_a[interval, ] %*% temp %*% XTVinvX_a[interval, ]
        #partC[i,] = XTVinvX_a[interval, ] %*% temp %*% XTVinvX_a[interval, ]
        # part (D) have 2 data sets.    has beta
        partD[i,] = - X_aVY[i, ]
        # part (E)
        partE[i,] = X_aVX_a[i]
        #print(i)
      }   
    }else{
    for (i in 1:Nparam){
      interval <- c(((i-1)*Ucov+1) : (i*Ucov))
      eigenH = eigen(XTVinvX_deleted[interval, ])
      eig = list(values=1/eigenH$values, vectors=eigenH$vectors)
      temp <- eig$vectors %*% diag(eig$values) %*% t(eig$vectors)
      #temp <- solve(XTVinvX_deleted[interval, ]) 
      #temp %*% XTVinvX_deleted[interval, ]
      # part (A) have 2 data sets
      partA[i,] = XVY_deleted[interval,] %*% temp %*% XVY_deleted[interval,]
      # part (B) have 2 data sets.   has beta
      partB[i,] = - XTVinvX_a[interval,] %*% temp %*% XVY_deleted[interval,]
      # part (C) no data sets.  has beta
      partC[i,] = XTVinvX_a[interval, ] %*% temp %*% XTVinvX_a[interval, ]
      #partC[i,] = XTVinvX_a[interval, ] %*% temp %*% XTVinvX_a[interval, ]
      # part (D) have 2 data sets.    has beta
      partD[i,] = - X_aVY[i, ]
      # part (E)
      partE[i,] = X_aVX_a[i]
      #print(i)
    }
    }
  }else{
    for (i in 1:Nparam){
      interval <- c(((i-1)*Ucov+1) : (i*Ucov))
      temp <- solve(XTVinvX_deleted[interval, ]) 
      if(Ucov==1){
        #diag(XVY_deleted[interval,] %*% temp  %*% XVY_deleted[interval,])
        partA[i,] = rowSums(XVY_deleted[interval,] %*% temp * XVY_deleted[interval,])
        # part (B) have 2 data sets.   has beta
        partB[i,] = -(XTVinvX_a[interval,]) %*% temp %*% XVY_deleted[interval,]
        # part (C) no data sets.  has beta
        partC[i,] = XTVinvX_a[interval, ] %*% temp %*% XTVinvX_a[interval, ]
        # part (D) have 2 data sets.    has beta
        partD[i,] = - X_aVY[i, ]
        # part (E)
        partE[i,] = X_aVX_a[i]
      }else{
        #diag(t(XVY_deleted[interval,]) %*% temp %*% XVY_deleted[interval,])
        # part (A) have 2 data sets
        partA[i,] = rowSums(t(XVY_deleted[interval,]) %*% temp * t(XVY_deleted[interval,]))
        # part (B) have 2 data sets.   has beta
        partB[i,] = -(XTVinvX_a[interval,]) %*% temp %*% XVY_deleted[interval,]
        # part (C) no data sets.  has beta
        partC[i,] = XTVinvX_a[interval, ] %*% temp %*% XTVinvX_a[interval, ]
        # part (D) have 2 data sets.    has beta
        partD[i,] = - X_aVY[i, ]
        # part (E)
        partE[i,] = X_aVX_a[i]
      }
    }
  }
  
  
  

  
# loglikAll = array(NA, c(m,Nparam,Ndata))
# 
# for (bet in 1:m){
#   ssqResidual <- ssqY + 2* BetaSlice[bet] *partD + BetaSlice[bet]^2 *partE - (partA + 2*BetaSlice[bet]* partB + BetaSlice[bet]^2 * partC)
#   loglik_forthisbeta <- (-0.5)*(Nobs*log(ssqResidual/Nobs) + detVar + Nobs + Nobs*log(2*pi) + jacobian)
#   
#     loglikAll[bet,,] = loglik_forthisbeta
# 
#     #LogLik_optimized[bet,] = max(loglik_forthisbeta)
# }
# 
# 
#    aaa<-matrix(loglikAll[,,1], nrow=m, ncol=Nparam)
#    matplot(BetaSlice, aaa , type='l', xlab='intercept', lty=1,  col='#00000040', ylim=c(-350, ))
#    abline(v=c(simRes$summary['(Intercept)',c('estimate','ci0.1','ci0.9')]))
#    breaks[a] <- max(aaa) - qchisq(cilevel,  df = 1)/2
# #  lines(BetaSlice, LogLik_optimized, col='red')
#    plot(BetaSlice, LogLik_optimized[,1], lwd=2, type='l',col='red')
#    lines(BetaSlice, loglikAll[,1,1], col='green', lwd=2, type='l')
#   lines(BetaSlice, loglikAll[,1,17], col='blue', lwd=2, type='l')   #, ylim = c(-50,0) + max(loglikAll)
#   lines(BetaSlice, loglikAll[,1,1], col='green', lwd=2, type='l')
#   lines(BetaSlice, loglikAll[,5979,1], col='black', lwd=2, type='l')
#   lines(BetaSlice, loglikAll[,149,32], col='blue', lwd=2, type='l')
#   lines(BetaSlice, loglikAll[,7246,32], col='blue', lwd=2, type='l')
#   lines(BetaSlice, loglikAll[,6967,32], col='blue', lwd=2, type='l')
#   lines(BetaSlice, loglikAll[,7246,1], col='yellow', lwd=2, type='l')
#   lines(BetaSlice, loglikAll[,6268,1], col='blue', lwd=2, type='l')
#   lines(BetaSlice, loglikAll[,2330,1], col='blue', lwd=2, type='l')
#   lines(BetaSlice, loglikAll[,7246,32], col='blue', lwd=2, type='l')
# 
#   lines(BetaSlice, loglikAll[,878,32], col='green', lwd=2, type='l'), ylim = c(-50,0) + max(loglikAll))

#   abline(h=breaks[a], col='blue')
#   result$params[c(1,5979),]
# bestParam =   which.max(apply(loglikAll, c(1,3), max))
# bestBocxox = which.max(apply(loglikAll[,bestParam,], 2, max))
# lines(Betas, loglikAll[,bestParam, bestBocxox], col='blue')
if(reml==FALSE){
  for (bet in 1:m){
    ssqResidual <- ssqY + 2* BetaSlice[bet] *partD + BetaSlice[bet]^2 *partE - (partA + 2*BetaSlice[bet]* partB + BetaSlice[bet]^2 * partC)
    loglik_forthisbeta <- (-0.5)*(Nobs*log(ssqResidual/Nobs) + detVar + Nobs + Nobs*log(2*pi) + jacobian)
    # if(a==1){
    # index[bet,] <- which(loglik_forthisbeta == max(loglik_forthisbeta, na.rm = TRUE), arr.ind = TRUE)
    # }
    if(!is.null(I)){
    LogLik_optimized[bet,1] = max(loglik_forthisbeta[,]) 
    }else{
    LogLik_optimized[bet,a] = max(loglik_forthisbeta[,])
    }
  }
}else if(reml==TRUE){
  for (bet in 1:m){
    ssqResidual <- ssqY + 2* BetaSlice[bet] *partD + BetaSlice[bet]^2 *partE - (partA + 2*BetaSlice[bet]* partB + BetaSlice[bet]^2 * partC)
    loglik_forthisbeta <- (-0.5)*((Nobs-Ncov)*log(ssqResidual/(Nobs-Ncov)) + detVar +detReml+ jacobian + Nobs*log(2*pi) + Nobs-Ncov)
    LogLik_optimized[bet,a] = max(loglik_forthisbeta[,])
  }  
}  
  # result$params
  
  

  
  # Sconfigs = c(1, seq(4*608, 608*10, len=50))
  # xx <- matrix(0, nrow=m, ncol=length(Sconfigs))
  # par(mfrow = c(1,1), mar = c(3,3, 0, 0))
  # for (bet in 1:m){
  #   ssqResidual <- ssqY + BetaSlice[bet] *partD + BetaSlice[bet]^2 *partE - (partA + BetaSlice[bet]* partB + BetaSlice[bet]^2 * partC)
  #   loglik_forthisbeta <- (Nobs*log(ssqResidual/Nobs) + detVar + Nobs + Nobs*log(2*pi) + jacobian)*(-0.5)
  #   loglik_forthisbeta <- loglik_forthisbeta[Sconfigs]
  #   
  #   xx[bet,] <- loglik_forthisbeta #[order(loglik_forthisbeta,decreasing = TRUE)]#[1:100]]
  # }
  # which (loglik_forthisbeta == min(loglik_forthisbeta))
  
#   matplot(BetaSlice, xx, type='l', xlab='intercept', lty=1, col=c("#FF0000", rep("#00000070", length(Sconfigs)-1)),
#           ylim = max(xx) + c(-6, 0))#, ylim = max(loglik_forthisbeta) + c(-1, 0))
#   lines(BetaSlice, xx[,1], col='red')
#        #lines(BetaSlice, loglik_forthisbeta[])
#   loglik_forthisbeta[order(loglik_forthisbeta,decreasing = TRUE)[1:100]]
  #  aaa<-matrix(loglikAll[,,12], nrow=m, ncol=Nparam)
  #  matplot(BetaSlice, aaa, type='l', xlab='intercept', lty=1, ylim=c(-395,-377),   col='#00000040')
  #  abline(v=c(simRes$summary['(Intercept)',c('estimate','ci0.1','ci0.9')]))
  #  lines(BetaSlice, LogLik_optimized, col='red')
  #  plot(BetaSlice, LogLik_optimized, col='red')
  #  dim(aaa)
  # 
  # aaa[,1:3]
  
  
    ############### ci ###########################################
    lower = min(BetaSlice)
    upper = max(BetaSlice)
    if(!is.null(I)){
    LogLik <- LogLik_optimized[,1]        
    }else{
    #which.max(LogLik_optimized[,a])
    LogLik <- LogLik_optimized[,a]      
    }
    index<-which.max(LogLik)
    #f1 <- splinefun(Betas, LogLik, method = "fmm")
    breaks[a] <- max(LogLik) - qchisq(cilevel,  df = 1)/2
    plot(BetaSlice, LogLik-breaks[a],  ylim = max(LogLik-breaks[a]) + c(-3, 0.2), xlim = range(BetaSlice[max(LogLik-breaks[a]) - LogLik+breaks[a] < 3]), cex=0.2, xlab=paste('beta',a), ylab="profileL", col='blue')
    abline(h=0, lty = 2, col='black')
    
    # temp <- qchisq(cilevel,  df = 1)/2
    # plot(BetaSlice*1e03, LogLik-breaks[a]-temp,  ylim = max(LogLik-breaks[a]-temp) + c(-3, 0.2), xlim = range(BetaSlice[max(LogLik-breaks[a]-temp) - LogLik+breaks[a]+temp < 3]*1e03), cex=0.2, xlab='elevation*1000', col='blue', ylab='profileLogL')
    # abline(h=-temp, lty = 2, col='black')
    
    
    
    
    if(convexHull == TRUE){
    profileLogLik <- as.data.frame(cbind(BetaSlice, LogLik-breaks[a]))
    colnames(profileLogLik) <- c("x1",'profile')
    datC2 = geometry::convhulln(profileLogLik)
    allPoints = unique(as.vector(datC2))
    toTest = profileLogLik[allPoints,]
    toTest[,'profile'] = toTest[,'profile'] + 0.1
    inHull = geometry::inhulln(datC2, as.matrix(toTest))
    toUse = profileLogLik[allPoints,][!inHull,]
    toTest = profileLogLik[allPoints,]
    toUse <- toUse[order(toUse$x1),]
    
    points(toTest, col='red', cex=0.6)
    points(toUse, col='blue', cex=0.6, pch=3)
    
    
    interp1 = mgcv::gam(profile ~ s(x1, k=nrow(toUse),  m=1, fx=TRUE), data=toUse)
    prof = data.frame(x1=seq(min(toUse$x1), max(toUse$x1), len=1001))
    prof$z = predict(interp1, prof)

    #lines(prof$x1, prof$z, col = 'black')
    f1 <- approxfun(toUse[,1], toUse[,2])
    curve(f1(x), add = TRUE, col = 2, n = 1001)
    profBetas[,c((2*a-1):(2*a))] <- as.matrix(prof)
    }else if(convexHull == FALSE){
      f1 <- approxfun(BetaSlice, LogLik-breaks[a])
      curve(f1(x), add = TRUE, col = 2, n = 1001)
      # BetaSlice2 <- sort(BetaSlice)
      # cc <- (LogLik-breaks[a])[order(BetaSlice,decreasing = F)]
      # lines(BetaSlice2, cc,col='green')
      #plot(BetaSlice,LogLik-breaks[a], cex=0.2)
    }
    MLE <- BetaSlice[index]
    #result <- optimize(f1, c(lower, upper), maximum = TRUE, tol = 0.0000000001)  # added 3 more 0s to get a accurate MLE for beta2 in swiss
    # result$maximum
    ci<-rootSolve::uniroot.all(f1, lower = lower, upper = upper)
    abline(v =c(MLE,ci), lty = 2, col='black')
    # text(MLE, -3, round(MLE,digits = 3))
    # text(ci[1], -3, round(ci[1],digits = 3))
    # text(ci[2], -3, round(ci[2],digits = 3))

    #abline(h=breaks[a], lty = 2)
    #f2 <- splinefun(Betas, LogLik-breaks[a], method = "fmm")
    #f2 <- approxfun(Betas, LogLik-breaks[a])
    #plot(Betas,LogLik-breaks[a])
    #curve(f2(x), add = TRUE, col = 2, n = 1001)
    # ci <- rootSolve::uniroot.all(f1, lower = lower, upper = upper)

    if(length(ci)==1){
      if( ci > MLE){
      ci <- c(lower, ci)
      }else{
      ci <- c(ci, upper)}
    }
    
    if(length(ci)==0){
      ci <- c(lower, upper)
    }
    
    if(length(ci)>2){
    warning("invalid ci returned")
    ci <- c(NA, NA)
    }
    ############### output #####################################
    Table[a,] <- c(MLE, ci, max(LogLik))
    
  }
  
  if(convexHull == TRUE){
    Output <- list(estimates = Table,
                   profBetas = profBetas,
                   LogLik = LogLik_optimized,
                   breaks = breaks)
  }else{
    Output <- list(estimates = Table,
                   LogLik = LogLik_optimized,
                   breaks = breaks)    
  }
    
    Output

  
}  
  
  
  
  
