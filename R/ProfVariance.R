#' @title Estimate Log-likelihood for Gaussian random fields when sdSpatial are given
#' @import data.table
#' @useDynLib gpuLik
#' @export


######## this function will be incorporated in the other function later i think

       profVariance<- function(sdSpatial, #a vector  given by the user 
                               cilevel=0.95,
                               Nobs, 
                               Ndata,
                               Nparam,
                               Ncov,
                               detVar, 
                               detReml,
                               ssqResidual, 
                               jacobian, 
                               reml=FALSE,
                               convexHull = FALSE){
  
  

  detVar <- matrix(rep(detVar, Ndata), nrow=Nparam)
  jacobian <- do.call(rbind, replicate(Nparam, jacobian, simplify = FALSE))
  m <- length(sdSpatial)
  LogLik = matrix(0, nrow=m, ncol=1)
  
  
if(reml==FALSE){
  for (var in 1:m){
    All_min2loglik_forthisvar <- ssqResidual/(sdSpatial[var]^2) + Nobs*log(sdSpatial[var]^2) + detVar + jacobian + Nobs*log(2*pi) 
    LogLik[var,] <- -0.5*min(All_min2loglik_forthisvar)
  }
}else if(reml==TRUE){
  for (var in 1:m){
    All_min2loglik_forthisvar <- ssqResidual/(sdSpatial[var]^2) + (Nobs-Ncov)*log(sdSpatial[var]^2) + detVar +detReml + jacobian + Nobs*log(2*pi) 
    LogLik[var,] <- -0.5*min(All_min2loglik_forthisvar)
  }  
}
  
  lower = min(sdSpatial)
  upper = max(sdSpatial)
  breaks <- max(LogLik) - qchisq(cilevel,  df = 1)/2
  plot(sdSpatial,LogLik-breaks)
  abline(h=0, lty = 2, col='black')
  if(convexHull == TRUE){
    profileLogLik <- as.data.frame(cbind(sdSpatial, LogLik-breaks))
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
    profsdSpatial <- as.matrix(prof)
  }else if(convexHull == FALSE){
  #f1 <- splinefun(sdSpatial, LogLik, method = "monoH.FC")
  f1 <- approxfun(sdSpatial, LogLik-breaks)
  curve(f1(x), add = TRUE, col = 2, n = 1001)
  }
  #result <- optimize(f1, c(lower, upper), maximum = TRUE, tol = 0.000001)
  MLE <- sdSpatial[which.max(LogLik)]
  #abline(h=breaks)
  #f2 <- splinefun(sdSpatial, LogLik-breaks, method = "monoH.FC")
  #f2 <- approxfun(sdSpatial, LogLik-breaks)
  #plot(sdSpatial,LogLik-breaks)
  #curve(f2(x), add = TRUE, col = 2, n = 1001)
  ci <- rootSolve::uniroot.all(f1, lower = lower, upper = upper)
  abline(v =c(MLE,ci), lty = 2, col='black')
  if(length(ci)==1){
    if( ci > MLE){
      ci <- c(lower, ci)
    }else{
      ci <- c(ci, upper)}
  }
  if(length(ci)==0){
    warning("require a better param matrix")
    ci <- c(NA, NA)
  }
  if(length(ci)>2){
    warning("invalid ci's retruned")
    ci <- ci[3:4]
  }
  
  ############### output #####################################
  Table <- matrix(NA, nrow=1, ncol=4)
  colnames(Table) <-  c("MLE",paste(c('lower', 'upper'), cilevel*100, 'ci', sep = ''),"maximum")
  Table[1,] <- c(MLE, ci, max(LogLik))
  
  if(convexHull == TRUE){
    Output <- list(estimates = Table,
                   profsdSpatial = profsdSpatial,
                   LogLik = LogLik,
                   breaks = breaks)
  }else{
  Output <- list(estimates = Table,
                 LogLik = LogLik,
                 breaks = breaks)
  }
  Output
  
}

       

       # result = cbind(sdSpatial, LogLik_optimized)
       # colnames(result) <- c("sdSpatial",'LogL')
       # MLE <- result[,'sdSpatial'][which.max(result[,'LogL'])]
       # # plot(result[,'sdSpatial'], result[,'LogL'])
       # maximum <- max(LogLik_optimized)
       # breaks95 = maximum - qchisq(0.95,  df = 1)/2
       # 
       # 
       # leftOfMax = result[,'sdSpatial'] < MLE
       # if(length(which(leftOfMax)) <=2 | length(which(!leftOfMax)) <=2){
       #   ci95 = c(NA,NA)
       #   print("Not enough data for CI calculation")
       # }else{
       #   afLeft <- approxfun(result[,'LogL'][leftOfMax], result[,'sdSpatial'][leftOfMax])   
       #   afRight <- approxfun(result[,'LogL'][!leftOfMax], result[,'sdSpatial'][!leftOfMax])   
       #   
       #   ci95= c(afLeft(breaks95), afRight(breaks95))
       # }
       
       
       # Output = list(LogLik_optimized=LogLik_optimized,
       #               ci95=ci95
       # )
       # 
       # Output
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       