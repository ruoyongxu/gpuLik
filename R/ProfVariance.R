#' @title Estimate Log-likelihood for Gaussian random fields when sdSpatial are given
#' @import data.table
#' @useDynLib gpuLik
#' @export


######## this function will be incorporated in the other function later i think

       profVariance<- function(sdSpatial, #a vector  given by the user 
                               cilevel=0.95,
                               Nobs, 
                               Nparam,
                               Ndata,
                               detVar, 
                               detReml,
                               ssqResidual, 
                               jacobian, 
                               reml=FALSE){
  
  

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
  #f1 <- splinefun(sdSpatial, LogLik, method = "monoH.FC")
  breaks <- max(LogLik) - qchisq(cilevel,  df = 1)/2
  f1 <- approxfun(sdSpatial, LogLik-breaks)
  plot(sdSpatial,LogLik-breaks)
  curve(f1(x), add = TRUE, col = 2, n = 1001)
  
  #result <- optimize(f1, c(lower, upper), maximum = TRUE, tol = 0.000001)
  MLE <- sdSpatial[which.max(LogLik)]
  #abline(h=breaks)
  #f2 <- splinefun(sdSpatial, LogLik-breaks, method = "monoH.FC")
  #f2 <- approxfun(sdSpatial, LogLik-breaks)
  #plot(sdSpatial,LogLik-breaks)
  #curve(f2(x), add = TRUE, col = 2, n = 1001)
  abline(h=0, lty = 2, col='black')
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
  
  
  Output <- list(estimates = Table,
                 LogLik = LogLik,
                 breaks = breaks)
  
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
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       