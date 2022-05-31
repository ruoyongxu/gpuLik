#' @title Estimate profile Log-likelihood for covariance parameters and lambda
#' @import data.table
#' @importFrom rootSolve uniroot.all
#' @useDynLib gpuLik



# get1dCovexhull <- function(profileLogLik,     # a data frame or data.table # 2 column names must be x1 and profile
#                                 a=0.1,    # minus a little thing
#                                 b=0,
#                                 m=1,
#                                 seqvalue){
#   
#   datC2 = geometry::convhulln(profileLogLik)
#   allPoints = unique(as.vector(datC2))
#   toTest = profileLogLik[allPoints,]
#   toTest[,'profile'] = toTest[,'profile'] + a
#   inHull = geometry::inhulln(datC2, as.matrix(toTest))
#   toUse = profileLogLik[allPoints,][!inHull,]
#   toTest = profileLogLik[allPoints,]
#   
#   # datC1= geometry::convhulln(profileLogLik)
#   # allPoints1 = unique(as.vector(datC1))
#   # toTest = profileLogLik[allPoints1,]
#   # toTest[,'profile'] = toTest[,'profile'] - a
#   # toTest[,'x1'] = toTest[,'x1'] + b
#   # inHull1 = geometry::inhulln(datC1, as.matrix(toTest))
#   # toUse = profileLogLik[allPoints1,][inHull1,]
#   # toTest[,'profile'] = toTest[,'profile'] + a
#   # toTest[,'x1'] = toTest[,'x1'] - b
#   
#   interp1 = mgcv::gam(profile ~ s(x1, k=nrow(toUse), m=m, fx=TRUE), data=toUse)
#   prof1 = data.frame(x1=seq(seqvalue[1], seqvalue[2], len=201))
#   prof1$z = predict(interp1, prof1)
#   
#   output <- list(toUse = toUse, toTest = toTest, prof=prof1)
#   
#   output
# }








#' @export
# betahat and sigmahat 
# profile log-likelihood for each covariance parameters + lambda
   simLgmCov1d <- function(data,
                           formula, 
                           coordinates,
                           params, # CPU matrix for now, users need to provide proper parameters given their specific need
                           paramToEstimate = c('range','nugget'),
                           boxcox,  # boxcox is always estimated
                           cilevel,  # decimal
                           type = c("float", "double")[1+gpuInfo()$double_support],
                           reml=FALSE, 
                           NparamPerIter,
                           Nglobal,
                           Nlocal,
                           NlocalCache,
                           verbose=c(1,0)){
  
  
  
  if(1 %in% boxcox){
    HasOne=TRUE
  }else{
    HasOne=FALSE 
  }
  
  
  if(0 %in% boxcox){
    boxcox = c(1, 0, setdiff(boxcox, c(1,0)))
  }else{
    # will always be c(1,.......)
    boxcox = c(1, setdiff(boxcox, 1))
  }
  
  
  
  # get rid of NAs in data
  data = data.frame(data)
  theNA = apply(  data[,all.vars(formula),drop=FALSE],
                  1, 
                  function(qq) any(is.na(qq))
  )
  noNA = !theNA
  
  
  ############## the X matrix #################################
  covariates = model.matrix(formula, data[noNA,])
  observations = all.vars(formula)[1]
  observations = data.matrix(data[noNA, observations, drop=FALSE])
  
  Nobs = nrow(covariates)
  Ncov = ncol(covariates)
  Ndata = length(boxcox)
  Nparam = nrow(params)
  
  # whole data set including columns for transformed y's
  yx = vclMatrix(cbind(observations, matrix(0, Nobs, Ndata-1), covariates), 
                 type=type)
  
  
  # coordinates
  coordsGpu = vclMatrix(coordinates[noNA,], type=type)  
  # box-cox
  boxcoxGpu = vclVector(boxcox, type=type)
  
  # prepare params, make sure variance=1 in params
  params0 = geostatsp::fillParam(params)
  params0[,"variance"]=1 
  paramsGpu = vclMatrix(cbind(params0, matrix(0, nrow(params0), 22-ncol(params0))),type=type)
  
  varMat = vclMatrix(0, Nobs*NparamPerIter, Nobs, type=type)
  cholDiagMat = vclMatrix(0, NparamPerIter, Nobs, type=type)
  ssqY <- vclMatrix(0, Nparam, Ndata, type=type)
  XVYXVX = vclMatrix(0, Nparam * Ncov, ncol(yx), type=type)
  ssqBetahat <- vclMatrix(0, Nparam, Ndata, type=type)
  detVar = vclVector(0, Nparam,type=type)
  detReml = vclVector(0, Nparam, type=type)
  jacobian = vclVector(0, Ndata, type=type)   
  ssqYX = vclMatrix(0, ncol(yx) * NparamPerIter, ncol(yx), type=type)
  ssqYXcopy = vclMatrix(0, ncol(yx) * NparamPerIter, ncol(yx), type=type)
  LinvYX = vclMatrix(0, Nobs * NparamPerIter, ncol(yx), type=type)
  QinvSsqYx = vclMatrix(0, NparamPerIter*Ncov, Ndata, type = type)
  cholXVXdiag = vclMatrix(0, NparamPerIter, Ncov, type=type)
  minusTwoLogLik <- vclMatrix(0, Nparam, Ndata, type=type)
  
  
  gpuLik:::likfitGpu_BackendP(
    yx,        #1
    coordsGpu, #2
    paramsGpu, #3
    boxcoxGpu,   #4#betasGpu,  #5
    ssqY,     #5#aTDinvb_beta,
    XVYXVX,
    ssqBetahat, #7#ssqBeta,
    detVar,    #11
    detReml,   #12
    jacobian,  #13
    NparamPerIter,  #14
    as.integer(Nglobal),  #12
    as.integer(Nlocal),  #16
    NlocalCache,  #14
    verbose=verbose,  #15
    ssqYX, #
    ssqYXcopy,  #new
    LinvYX,  #18
    QinvSsqYx, 
    cholXVXdiag, #20
    varMat,        #21     Vbatch
    cholDiagMat)
  
  # resid^T V^(-1) resid, resid = Y - X betahat = ssqResidual
  ssqResidual <- ssqY - ssqBetahat
  
  # params0[which(is.na(as.vector(detVar))),]
  
  
  if(reml== FALSE){ 
    # if(fixVariance == FALSE){ # ml
    gpuLik:::matrix_vector_sumBackend(Nobs*log(ssqResidual/Nobs),
                                         detVar,
                                         jacobian,  
                                         Nobs + Nobs*log(2*pi),
                                         minusTwoLogLik,
                                         Nglobal)
    
  }else if(reml==TRUE){
    # if(fixVariance == FALSE){# remlpro
    # minusTwoLogLik= (n-p)*log(ssqResidual/(n-p)) + logD + logP + jacobian + n*log(2*pi) + n-p    
    gpuLik:::matrix_vector_sumBackend((Nobs-Ncov)*log(ssqResidual/(Nobs-Ncov)),
                                         detVar+detReml,
                                         jacobian,  
                                         Nobs*log(2*pi)+Nobs-Ncov,
                                         minusTwoLogLik,
                                         Nglobal)
    
  }
  
  
  LogLikcpu <- as.matrix(-0.5*minusTwoLogLik)
  colnames(LogLikcpu) <- paste(c('boxcox'), round(boxcox, digits = 3) ,sep = '')
  selected_rows <- which(is.na(as.vector(detVar)))
  if(length(selected_rows)==0){
    paramsRenew <- params
    detVar2 <- as.vector(detVar)
    detReml2 <- as.vector(detReml)
    ssqY2 <- as.matrix(ssqY)
    ssqBetahat2 = as.matrix(ssqBetahat)
    ssqResidual2 = as.matrix(ssqResidual)
    XVYXVX2 <- as.matrix(XVYXVX)
  }else{
    Nparam = Nparam - length(selected_rows)
    paramsRenew <- params[-selected_rows,]
    LogLikcpu <- as.matrix(LogLikcpu[-selected_rows,]) 
    detVar2 <- as.vector(detVar)[-selected_rows]
    detReml2 <- as.vector(detReml)[-selected_rows]
    ssqY2 <- as.matrix(ssqY)[-selected_rows,]
    ssqBetahat2 = as.matrix(ssqBetahat)[-selected_rows,]
    ssqResidual2 = as.matrix(ssqResidual)[-selected_rows,]
    XVYXVX2 <- as.matrix(XVYXVX)
    #XVYXVX3 <- as.matrix(XVYXVX)
    #which(is.na(XVYXVX2),arr.ind = TRUE)[,1]
    #which(is.na(XVYXVX3),arr.ind = TRUE)[,1]
    #tempp <- unique(which(is.na(XVYXVX2),arr.ind = TRUE)[,1])
    #tempp[-seq(1,length(tempp),2)]/2
    #unique(which(is.na(XVYXVX2),arr.ind = TRUE)[,1])
    
    a <- 0   
    for (j in 1:length(selected_rows)){
      a<-c(a, c(((selected_rows[j]-1)*Ncov+1): (selected_rows[j]*Ncov)))
    }
    a <- a[-1]
    XVYXVX2 <- XVYXVX2[-a,   ]
  }
  
  if(HasOne==FALSE){
    LogLikcpu <- as.matrix(LogLikcpu[,-1])
    ssqY2 <- ssqY2[,-1]
    ssqBetahat2 <- ssqBetahat2[,-1]
    ssqResidual2 <- ssqResidual2[,-1]
    Ndata <- length(boxcox) - 1
    XVYXVX2 <- XVYXVX2[,-1]
    jacobian <- as.vector(jacobian)[-1]
    boxcox <- boxcox[-1]
  }
  
  ############## output matrix ####################
  Table <- matrix(NA, nrow=length(paramToEstimate) + Ncov + 1, ncol=3)
  rownames(Table) <-  c(colnames(covariates), "sdSpatial", paramToEstimate)
  colnames(Table) <-  c("estimate", paste(c('lower', 'upper'), cilevel*100, 'ci', sep = ''))
  
  
  index <- which(LogLikcpu == max(LogLikcpu, na.rm = TRUE), arr.ind = TRUE)
  #################sigma hat#########################
  if(reml==FALSE)  {
    Table["sdSpatial",1] <- sqrt(ssqResidual[index[1],index[2]]/Nobs)
  }else{         
    Table["sdSpatial",1] <- sqrt(ssqResidual[index[1],index[2]]/(Nobs - Ncov))
  }
  
  maximum <- max(LogLikcpu)
  breaks = maximum - qchisq(cilevel,  df = 1)/2
  breaks2d = maximum - qchisq(cilevel,  df = 2)/2
  par(mfrow = c(3, 2))
  
  ############### profile for covariance parameters #####################
  aniso1 <-  unname(sqrt(paramsRenew[,'anisoRatio']-1) * cos(2*(paramsRenew[,'anisoAngleRadians'])))
  aniso2 <-  unname(sqrt(paramsRenew[,'anisoRatio']-1) * sin(2*(paramsRenew[,'anisoAngleRadians'])))
  aniso <- cbind(aniso1, aniso2)
  combinedRange <- sqrt(paramsRenew[,'range']^2/paramsRenew[,'anisoRatio'])
  paramsRenew <- cbind(paramsRenew, combinedRange, aniso, sqrt(paramsRenew[,"nugget"]) * Table["sdSpatial",1])
  colnames(paramsRenew)[ncol(paramsRenew)] <- 'sdNugget'
  
  Spars = c("range","combinedRange","nugget",'sdNugget',"shape",'aniso1','aniso2','anisoRatio','anisoAngleRadians')
  result = data.table::as.data.table(cbind(LogLikcpu, paramsRenew[,Spars]))
  profileLogLik <- result[, .(profile=max(.SD)), by=Spars]
  
  profileLogLik[,'profile'] <- profileLogLik[,'profile'] - breaks
  profileLogLik <- profileLogLik[profile > maximum- breaks-10]  #maximum- breaks 
  profileLogLik <- as.data.frame(profileLogLik)
  
  
  
  
  
  ######################range ########
  if('combinedRange' %in% paramToEstimate){
    plot(profileLogLik$combinedRange, profileLogLik$profile, log='x',cex=.4, xlab="combinedRange",pch=16, ylab="profileLogL")
    
    profileLogLik$sumLogRange <- 2*log(profileLogLik$combinedRange)
    newdata <- profileLogLik[,c('sumLogRange','profile')]
    colnames(newdata)[1]<-"x1"     
    
    datC2 = geometry::convhulln(newdata)
    allPoints = unique(as.vector(datC2))
    toTest = newdata[allPoints,]
    toTest[,'profile'] = toTest[,'profile'] + 0.1
    inHull = geometry::inhulln(datC2, as.matrix(toTest))
    toUse = newdata[allPoints,][!inHull,]
    toTest = newdata[allPoints,]
    
    
    #plot(profileLogLik$sumLogRange, profileLogLik$profile,cex=.4, xlab="sumLogRange",pch=16, ylab="profileLogL", col = colAlpha$plot)
    interp1 = mgcv::gam(profile ~ s(x1, k=nrow(toUse),  m=1, fx=TRUE), data=toUse)
    profsumLogRange = data.frame(x1=seq(min(toUse$x1), max(toUse$x1), len=1001))
    profsumLogRange$z = predict(interp1, profsumLogRange)
    
    points(exp(0.5*toTest[,1]), toTest[,2], col='red', cex=0.6)
    points(exp(0.5*toUse[,1]), toUse[,2], col='blue', cex=0.6, pch=3)
    lines(profsumLogRange$x1, profsumLogRange$z, col = 'green')
    lines(exp(0.5*profsumLogRange$x1), profsumLogRange$z, col = 'green')
    abline(h = 0, lty = 2, col='red')
    lower = min(newdata$x1)
    upper = max(newdata$x1)
    #f1 <- approxfun(profsumLogRange$x1, profsumLogRange$z)
    f1 <- approxfun(toUse[,1], toUse[,2])
    MLE <- sqrt(paramsRenew[index[1],'range']^2/paramsRenew[index[1],'anisoRatio'])
    ci<-rootSolve::uniroot.all(f1, lower = lower, upper = upper)
    abline(v = c(MLE,exp(0.5*ci)), lty = 2, col='red')
    if(length(ci)==1){
      if(ci > MLE){
        ci <- c(lower, ci)
        message("did not find lower ci for combinedRange")
      }else{
        ci <- c(ci, upper)
        message("did not find upper ci for combinedRange")}
    }
    
    if(length(ci)==0 | length(ci)>2){
      warning("error in paramsRenew")
      ci <- c(NA, NA)
    }
    Table["combinedRange",] <- c(MLE,exp(0.5*ci))
    
  }
  
  
  if('range' %in% paramToEstimate){
    plot(profileLogLik$range, profileLogLik$profile, log='x', cex=.2, xlab="range", ylab="profileLogL")
    
    profileLogLik$logrange <- log(profileLogLik$range)
    newdata <- profileLogLik[,c('logrange','profile')]
    colnames(newdata)[1]<-"x1"
    
    datC2 = geometry::convhulln(newdata)
    allPoints = unique(as.vector(datC2))
    toTest = newdata[allPoints,]
    toTest[,'profile'] = toTest[,'profile'] + 0.1
    inHull = geometry::inhulln(datC2, as.matrix(toTest))
    toUse = newdata[allPoints,][!inHull,]
    toTest = newdata[allPoints,]
    
    interp1 = mgcv::gam(profile ~ s(x1, k=nrow(toUse),  m=1, fx=TRUE), data=toUse)
    profrange = data.frame(x1=seq(min(toUse$x1), max(toUse$x1), len=1001))
    profrange$z = predict(interp1, profrange)
    
    
    points(exp(toTest[,1]),toTest[,2], col='red', cex=0.6)
    points(exp(toUse[,1]), toUse[,2], col='blue', cex=0.6, pch=3)
    
    lines(exp(profrange$x1), profrange$z, col = 'green')
    abline(h =0, lty = 2, col='red')
    lower = min(newdata$x1)
    upper = max(newdata$x1)
    #f1 <- approxfun(profrange$x1, profrange$z)
    f1 <- approxfun(toUse[,1], toUse[,2])
    MLE <- paramsRenew[index[1],'range']
    #MLE <- optimize(f1, c(lower, upper), maximum = TRUE, tol = 0.00000001)$maximum
    ci<-rootSolve::uniroot.all(f1, lower = lower, upper = upper)
    abline(v =c(MLE,exp(ci)), lty = 2, col='red')
    
    if(length(ci)==1){
      if(ci > MLE){
        ci <- c(lower, ci)
        message("did not find lower ci for range")
      }else{
        ci <- c(ci, upper)
        message("did not find upper ci for range")}
    }
    
    if(length(ci)==0 | length(ci)>2){
      warning("error in paramsRenew")
      ci <- c(NA, NA)
    }
    Table["range",] <- c(MLE,exp(ci))
  }
  
  
  ################shape ##############   
  if('shape' %in% paramToEstimate){
    plot(profileLogLik$shape, profileLogLik$profile, cex=.2, xlab="shape", ylab="profileLogL",  log='x')
    profileLogLik$logshape <- log(profileLogLik$shape)
    newdata <- profileLogLik[,c('logshape','profile')]
    colnames(newdata)[1]<-"x1"
    
    datC2 = geometry::convhulln(newdata)
    allPoints = unique(as.vector(datC2))
    toTest = newdata[allPoints,]
    toTest[,'profile'] = toTest[,'profile'] + 0.1
    inHull = geometry::inhulln(datC2, as.matrix(toTest))
    toUse = newdata[allPoints,][!inHull,]
    toTest = newdata[allPoints,]
    #toUse <- toUse[order(toUse$x1),]
    #toUse <- head(toUse, - 1)
    
    points(exp(toTest[,1]),toTest[,2], col='red', cex=0.6)
    points(exp(toUse[,1]), toUse[,2], col='blue', cex=0.6, pch=3)
    
    interp1 = mgcv::gam(profile ~ s(x1, k=nrow(toUse),  m=1, fx=TRUE), data=toUse)
    profShapeLog = data.frame(x1=seq(min(toUse$x1), max(toUse$x1), len=1001))
    profShapeLog$z = predict(interp1, profShapeLog)
    
    #plot(newdata$x1, newdata$profile, cex=.2, xlab="log(shape)", ylab="profileLogL")
    
    lines(exp(profShapeLog$x1), profShapeLog$z, col = 'green')
    abline(h =0, lty = 2, col='red') 
    #f1 <- approxfun(profShapeLog$x1, profShapeLog$z)
    f1 <- approxfun(toUse[,1], toUse[,2])
    
    lower = min(toUse$x1)
    upper = max(toUse$x1)
    MLE <- paramsRenew[index[1],'shape']
    #MLE <- optimize(f1, c(lower, upper), maximum = TRUE, tol = 0.00000001)$maximum
    ci<-rootSolve::uniroot.all(f1, lower = lower, upper = upper)
    abline(v =c(MLE,exp(ci)), lty = 2, col='red')
    if(length(ci)==1){
      if(ci > MLE){
        ci <- c(lower, ci)
        message("did not find lower ci for shape")
      }else{
        ci <- c(ci, upper)
        message("did not find upper ci for shape")}
    }
    
    if(length(ci)==0 | length(ci)>2){
      warning("error in paramsRenew")
      ci <- c(NA, NA)
    }
    Table["shape",] <- c(MLE,exp(ci))
  }       
  
  
  
  ################sd nugget ##############     
  if('sdNugget' %in% paramToEstimate){
    plot(profileLogLik$sdNugget, profileLogLik$profile, cex=.2, xlab="sdNugget", ylab="profileLogL")
    
    profileLogLik1 <- profileLogLik[,c('sdNugget','profile')]
    colnames(profileLogLik1) <- c("x1", 'profile')
    
    datC2 = geometry::convhulln(profileLogLik1)
    allPoints = unique(as.vector(datC2))
    toTest = profileLogLik1[allPoints,]
    toTest[,'profile'] = toTest[,'profile'] + 0.1
    inHull = geometry::inhulln(datC2, as.matrix(toTest))
    toUse = profileLogLik1[allPoints,][!inHull,]
    toTest = profileLogLik1[allPoints,]
    
    interp1 = mgcv::gam(profile ~ s(x1, k=nrow(toUse),  m=1, fx=TRUE), data=toUse)
    profsdNugget = data.frame(x1=seq(min(toUse$x1), max(toUse$x1), len=1001))
    profsdNugget$z = predict(interp1, profsdNugget)
    
    points(toTest, col='red', cex=0.6)
    points(toUse, col='blue', cex=0.6, pch=3)
    lines(profsdNugget$x1, profsdNugget$z, col = 'green')
    abline(h =0, lty = 2, col='red')
    lower = min(profileLogLik1$x1)
    upper = max(profileLogLik1$x1)
    #f1 <- approxfun(profsdNugget$x1, profsdNugget$z)
    f1 <- approxfun(toUse[,1], toUse[,2])
    
    MLE <- sqrt(paramsRenew[index[1],"nugget"]) * Table["sdSpatial",1]
    #MLE <- optimize(f1, c(lower, upper), maximum = TRUE, tol = 0.00000001)$maximum
    ci<-rootSolve::uniroot.all(f1, lower = lower, upper = upper)
    abline(v =c(MLE,ci), lty = 2, col='red')
    
    if(length(ci)==1){
      if(ci > MLE){
        ci <- c(lower, ci)
        message("did not find lower ci for sdNugget")
      }else{
        ci <- c(ci, upper)
        message("did not find upper ci for sdNugget")}
    }
    
    if(length(ci)==0 | length(ci)>2){
      warning("error in paramsRenew")
      ci <- c(NA, NA)
    }
    Table["sdNugget",] <- c(MLE, ci)
    
  }
  
  
  
  if('nugget' %in% paramToEstimate){
    plot(profileLogLik$nugget, profileLogLik$profile, cex=.2, xlab="nugget", ylab="profileLogL")
    profileLogLik1 <- profileLogLik[,c('nugget','profile')]
    colnames(profileLogLik1) <- c("x1", 'profile')
    
    datC2 = geometry::convhulln(profileLogLik1)
    allPoints = unique(as.vector(datC2))
    toTest = profileLogLik1[allPoints,]
    toTest[,'profile'] = toTest[,'profile'] + 0.1
    inHull = geometry::inhulln(datC2, as.matrix(toTest))
    toUse = profileLogLik1[allPoints,][!inHull,]
    toTest = profileLogLik1[allPoints,]
    toUse <- toUse[order(toUse$x1),]
    MLE <- paramsRenew[index[1],'nugget']
    if(nrow(toUse)>2){
      
      interp1 = mgcv::gam(profile ~ s(x1, k=nrow(toUse), m=1, fx=TRUE), data=toUse)
      profNugget = data.frame(x1=seq(min(toUse$x1), max(toUse$x1), len=1001))
      profNugget$z = predict(interp1, profNugget)
      
      points(toTest, col='red', cex=0.6)
      points(toUse, col='blue', cex=0.6, pch=3)
      lines(profNugget$x1, profNugget$z, col = 'green')
      abline(h =0, lty = 2, col='red')
      #f1 <- approxfun(profNugget$x1, profNugget$z)
      f1 <- approxfun(toUse[,1], toUse[,2])
      
      lower = min(profileLogLik1$x1)
      upper = max(profileLogLik1$x1)
      #MLE <- optimize(f1, c(lower, upper), maximum = TRUE, tol = 0.00000001)$maximum
      ci<-rootSolve::uniroot.all(f1, lower = lower, upper = upper)
      abline(v =c(MLE,ci), lty = 2,  col='red')
    }else{
      profNugget = data.frame(x1=seq(min(toUse$x1), max(toUse$x1), len=1001))
      points(toTest, col='red', cex=0.6)
      points(toUse, col='blue', cex=0.6, pch=3)
      f1 <- approxfun(toUse$x1, toUse$profile)
      profNugget$z = f1(profNugget$x1)
      lines(toUse$x1, toUse$profile, col = 'green')
      abline(h =0, lty = 2, col='red')
      lower = min(profileLogLik$x1)
      upper = max(profileLogLik$x1)
      #MLE <- optimize(f1, c(lower, upper), maximum = TRUE, tol = 0.00000001)$maximum
      ci<-rootSolve::uniroot.all(f1, lower = lower, upper = upper)
      abline(v =c(0,ci), lty = 2,  col='red')
    }
    if(length(ci)==1){
      if(ci > MLE){
        ci <- c(lower, ci)
        message("did not find lower ci for nugget")
      }else{
        ci <- c(ci, upper)
        message("did not find upper ci for nugget")}
    }
    
    if(length(ci)==0 | length(ci)>2){
      warning("error in paramsRenew")
      ci <- c(NA, NA)
    }
    Table["nugget",] <- c(MLE, ci)
  }
  
  
  
  if('aniso1' %in% paramToEstimate){
    plot(profileLogLik$aniso1, profileLogLik$profile, cex=.2, xlab="aniso1", ylab="profileLogL")
    profileLogLik1 <- profileLogLik[,c('aniso1','profile')]
    colnames(profileLogLik1) <- c("x1", 'profile')
    
    datC2 = geometry::convhulln(profileLogLik1)
    allPoints = unique(as.vector(datC2))
    toTest = profileLogLik1[allPoints,]
    toTest[,'profile'] = toTest[,'profile'] + 0.1
    inHull = geometry::inhulln(datC2, as.matrix(toTest))
    toUse = profileLogLik1[allPoints,][!inHull,]
    toTest = profileLogLik1[allPoints,]
    
    points(toTest[,1], toTest[,2], col='red', cex=0.6)
    points(toUse[,1], toUse[,2], col='blue', cex=0.6, pch=3)
    
    interp1 = mgcv::gam(profile ~ s(x1, k=nrow(toUse), m=1, fx=TRUE), data=toUse)
    profaniso1 = data.frame(x1=seq(min(toUse$x1), max(toUse$x1), len=1001))
    profaniso1$z = predict(interp1, profaniso1)
    
    lines(profaniso1$x1, profaniso1$z, col = 'green')
    abline(h =0, lty = 2, col='red')
    #f1 <- approxfun(profaniso1$x1, profaniso1$z)
    f1 <- approxfun(toUse[,1], toUse[,2])
    lower = min(profaniso1$x1)
    upper = max(profaniso1$x1)
    
    MLE <- sqrt(paramsRenew[index[1],'anisoRatio']-1) * cos(2*(paramsRenew[index[1],'anisoAngleRadians']))
    #MLE <- optimize(f1, c(lower, upper), maximum = TRUE, tol = 0.00000001)$maximum
    ci<-rootSolve::uniroot.all(f1, lower = lower, upper = upper)
    abline(v =c(MLE,ci), lty = 2, col='red')
    if(length(ci)==1){
      if(ci > MLE){
        ci <- c(lower, ci)
        message("did not find lower ci for aniso1")
      }else{
        ci <- c(ci, upper)
        message("did not find upper ci for aniso1")}
    }
    
    if(length(ci)==0 | length(ci)>2){
      warning("error in paramsRenew")
      ci <- c(NA, NA)
    }
    Table["aniso1",] <- c(MLE, ci)
    
  }
  
  
  
  if('aniso2' %in% paramToEstimate){
    plot(profileLogLik$aniso2, profileLogLik$profile, cex=.2, xlab="aniso2", ylab="profileLogL")
    
    profileLogLik1 <- profileLogLik[,c('aniso2','profile')]
    colnames(profileLogLik1) <- c("x1", 'profile')
    datC2 = geometry::convhulln(profileLogLik1)
    allPoints = unique(as.vector(datC2))
    toTest = profileLogLik1[allPoints,]
    toTest[,'profile'] = toTest[,'profile'] + 0.1
    inHull = geometry::inhulln(datC2, as.matrix(toTest))
    toUse = profileLogLik1[allPoints,][!inHull,]
    toTest = profileLogLik1[allPoints,]
    points(toTest[,1], toTest[,2], col='red', cex=0.6)
    points(toUse[,1], toUse[,2], col='blue', cex=0.6, pch=3)
    
    
    interp1 = mgcv::gam(profile ~ s(x1, k=nrow(toUse), m=1, fx=TRUE), data=toUse)
    profaniso2 = data.frame(x1=seq(min(toUse$x1), max(toUse$x1), len=1001))
    profaniso2$z = predict(interp1, profaniso2)
    
    #f1 <- approxfun(profaniso2$x1, profaniso2$z)
    f1 <- approxfun(toUse[,1], toUse[,2])
    
    lines(profaniso2$x1, profaniso2$z, col = 'green')
    abline(h = 0, lty = 2, col='black')
    lower = min(profileLogLik1$x1)
    upper = max(profileLogLik1$x1)
    
    MLE <- sqrt(paramsRenew[index[1],'anisoRatio']-1) * sin(2*(paramsRenew[index[1],'anisoAngleRadians']))
    #MLE <- optimize(f1, c(lower, upper), maximum = TRUE, tol = 0.00000001)$maximum
    ci<-rootSolve::uniroot.all(f1, lower = lower, upper = upper)
    abline(v =c(MLE,ci), lty = 2, col='black')
    if(length(ci)==1){
      if(ci > MLE){
        ci <- c(lower, ci)
        message("did not find lower ci for aniso2")
      }else{
        ci <- c(ci, upper)
        message("did not find upper ci for aniso2")}
    }
    
    if(length(ci)==0 | length(ci)>2){
      warning("error in paramsRenew")
      ci <- c(NA, NA)
    }
    Table["aniso2",] <- c(MLE, ci)     
  }
  
  
  if(('anisoRatio' %in% paramToEstimate)){    # &  ('anisoAngleRadians' %in% paramToEstimate)
    plot(profileLogLik$anisoRatio, profileLogLik$profile, cex=.2, xlab="anisoRatio", ylab="profileLogL")
    profileLogLik1 <- profileLogLik[,c('anisoRatio','profile')]
    colnames(profileLogLik1) <- c("x1", 'profile')
    
    datC2 = geometry::convhulln(profileLogLik1)
    allPoints = unique(as.vector(datC2))
    toTest = profileLogLik1[allPoints,]
    toTest[,'profile'] = toTest[,'profile'] + 0.1
    inHull = geometry::inhulln(datC2, as.matrix(toTest))
    toUse = profileLogLik1[allPoints,][!inHull,]
    toTest = profileLogLik1[allPoints,]
    
    points(toTest, col='red', cex=0.6)
    points(toUse, col='blue', cex=0.6, pch=3)
    
    interp1 = mgcv::gam(profile ~ s(x1, k=nrow(toUse), m=1, fx=TRUE), data=toUse)
    profRatio = data.frame(x1=seq(min(toUse$x1), max(toUse$x1), len=1001))
    profRatio$z = predict(interp1, profRatio)
    
    lines(profRatio$x1, profRatio$z, col='green')
    #f1 <- approxfun(profRatio$x1, profRatio$z)
    f1 <- approxfun(toUse[,1], toUse[,2])
    
    abline(h =0, lty = 2, col='red')
    lower = min(profileLogLik1$x1)
    upper = max(profileLogLik1$x1)
    
    MLE <- paramsRenew[index[1],'anisoRatio']
    #MLE <- optimize(f1, c(lower, upper), maximum = TRUE, tol = 0.00000001)$maximum
    ci<-rootSolve::uniroot.all(f1, lower = lower, upper = upper)
    abline(v =c(MLE,ci), lty = 2, col='red')
    if(length(ci)==1){
      if(ci > MLE){
        ci <- c(lower, ci)
        message("did not find lower ci for anisoRatio")
      }else{
        ci <- c(ci, upper)
        message("did not find upper ci for anisoRatio")}
    }
    
    if(length(ci)==0 | length(ci)>2){
      warning("error in paramsRenew")
      ci <- c(NA, NA)
    }
    Table["anisoRatio",] <- c(MLE, ci)
  }
  
  
  if('anisoAngleRadians' %in% paramToEstimate){
    plot(profileLogLik$anisoAngleRadians, profileLogLik$profile, cex=.2, xlab="anisoAngleRadians", ylab="profileLogL")
    
    profileLogLik1 <- profileLogLik[,c('anisoAngleRadians','profile')]
    colnames(profileLogLik1) <- c("x1", 'profile')
    
    datC2 = geometry::convhulln(profileLogLik1)
    allPoints = unique(as.vector(datC2))
    toTest = profileLogLik1[allPoints,]
    toTest[,'profile'] = toTest[,'profile'] + 0.1
    inHull = geometry::inhulln(datC2, as.matrix(toTest))
    toUse = profileLogLik1[allPoints,][!inHull,]
    toTest = profileLogLik1[allPoints,]
    
    points(toTest, col='red', cex=0.6)
    points(toUse, col='blue', cex=0.6, pch=3)
    
    interp1 = mgcv::gam(profile ~ s(x1, k=nrow(toUse), m=1, fx=TRUE), data=toUse)
    profRadians = data.frame(x1=seq(min(toUse$x1), max(toUse$x1), len=1001))
    profRadians$z = predict(interp1, profRadians)
    lines(profRadians$x1, profRadians$z, col='green')
    
    #f1 <- approxfun(profRadians$x1, profRadians$z)
    f1 <- approxfun(toUse[,1], toUse[,2])
    curve(f1(x), add = TRUE, col = 'green', n = 1001)
    abline(h =0, lty = 2, col='red')
    lower = min(profileLogLik1$x1)
    upper = max(profileLogLik1$x1)
    
    MLE <- paramsRenew[index[1],'anisoAngleRadians']
    #MLE <- optimize(f1, c(lower, upper), maximum = TRUE, tol = 0.00000001)$maximum
    ci<-rootSolve::uniroot.all(f1, lower = lower, upper = upper)
    abline(v =c(MLE,ci), lty = 2, col='red')
    if(length(ci)==1){
      if(ci > MLE){
        ci <- c(lower, ci)
        message("did not find lower ci for anisoAngleRadians")
      }else{
        ci <- c(ci, upper)
        message("did not find upper ci for anisoAngleRadians")}
    }
    
    if(length(ci)==0 | length(ci)>2){
      warning("error in paramsRenew")
      ci <- c(NA, NA)
    }
    Table["anisoAngleRadians",] <- c(MLE, ci)
  }
  
  
  
  
  
  
  ###############lambda hat#####################
  if(('boxcox'%in% paramToEstimate)  & length(boxcox)>5 ){
    likForboxcox = cbind(boxcox, apply(LogLikcpu, 2,  max) )
    f1 <- approxfun(likForboxcox[,1], likForboxcox[,2]-breaks)
    plot(likForboxcox[,1], likForboxcox[,2]-breaks, ylab= "proLogL", xlab='boxcox', cex=0.5)
    curve(f1(x), add = TRUE, col = 2, n = 1001)   #the number of x values at which to evaluate
    abline(h =0, lty = 2)
    
    lower = min(boxcox)
    upper = max(boxcox)
    MLE <- boxcox[index[2]]
    #MLE <- optimize(f1, c(lower, upper), maximum = TRUE, tol = 0.00000001)$maximum
    ci<-rootSolve::uniroot.all(f1, lower = lower, upper = upper)
    abline(v =c(MLE,ci), lty = 2)
    
    if(length(ci)==1){
      if( ci > MLE){
        ci <- c(lower, ci)
        message("did not find lower ci for boxcox")
      }else{
        ci <- c(ci, upper)
        message("did not find upper ci for boxcox")}
    }else if(length(ci)>2){
      warning("error in param matrix")
      ci <- c(NA, NA)
    }
    Table["boxcox",] <- c(MLE, ci)
    
  }else if(is.element('boxcox',paramToEstimate)  & length(boxcox) <= 5){
    message("boxcox: not enough values for interpolation!")
  }
  
  NcolTotal = Ndata + Ncov
  
  ###############betahat#####################
  Betahat <- matrix(0, nrow=Ncov, ncol=Ndata)
  a<-c( ((index[1]-1)*Ncov+1) : (index[1]*Ncov) )
  mat <- XVYXVX2[a,((Ndata+1):NcolTotal)]
  mat[upper.tri(mat)] <- mat[lower.tri(mat)]
  Betahat <- solve(mat) %*% XVYXVX2[a,index[2]]
  Table[colnames(covariates), 1] <- Betahat
  
  if(all(c('sdNugget','shape', 'aniso1') %in% paramToEstimate)){
    Output <- list(LogLik=LogLikcpu,
                   breaks = breaks,
                   breaks2d = breaks2d,
                   mleIndex = index,
                   summary = Table,
                   profcombinedRange = profcombinedRange,
                   profShapeLog = profShapeLog,
                   profNugget = profNugget,
                   profsdNugget = profsdNugget,
                   profaniso1 = profaniso1,
                   profaniso2 = profaniso2,
                   params = paramsRenew,
                   Infindex = selected_rows,
                   Nobs = Nobs,
                   Ncov = Ncov,
                   Ndata = Ndata,
                   Nparam = Nparam,
                   ssqY = ssqY2,     
                   ssqBetahat = ssqBetahat2,
                   ssqResidual = ssqResidual2,
                   detVar = as.vector(detVar2),   
                   detReml = as.vector(detReml2),   
                   jacobian = as.vector(jacobian),
                   XVYXVX = XVYXVX2)
  }
  
  if(!('shape' %in% paramToEstimate)){
    Output <- list(LogLik=LogLikcpu,
                   breaks = breaks,
                   breaks2d = breaks2d,
                   mleIndex = index,
                   summary = Table,
                   profcombinedRange = profcombinedRange,
                   profNugget = profNugget,
                   profaniso1 = profaniso1,
                   profaniso2 = profaniso2,
                   params = paramsRenew,
                   Infindex = selected_rows,
                   boxcox = boxcox,
                   Nobs = Nobs,
                   Ncov = Ncov,
                   Ndata = Ndata,
                   Nparam = Nparam,
                   ssqY = ssqY2,     
                   ssqBetahat = ssqBetahat2,
                   ssqResidual = ssqResidual2,
                   detVar = as.vector(detVar2),   
                   detReml = as.vector(detReml2),   
                   jacobian = as.vector(jacobian),
                   XVYXVX = XVYXVX2)  
    
  }
  
  
  if('anisoRatio' %in% paramToEstimate & ('anisoAngleRadians' %in% paramToEstimate)){
    Output <- list(LogLik=LogLikcpu,
                   breaks = breaks,
                   breaks2d = breaks2d,
                   mleIndex = index,
                   summary = Table,
                   #BetahatTable = x,
                   profcombinedRange = profcombinedRange,
                   profShapeLog = profShapeLog,
                   profNugget = profNugget,
                   profRatio = profRatio,
                   profRadians = profRadians,
                   params = paramsRenew,
                   Infindex = selected_rows,
                   boxcox = boxcox,
                   Nobs = Nobs,
                   Ncov = Ncov,
                   Ndata = Ndata,
                   Nparam = Nparam,
                   ssqY = ssqY2,     
                   ssqBetahat = ssqBetahat2,
                   ssqResidual = ssqResidual2,
                   detVar = as.vector(detVar2),   
                   detReml = as.vector(detReml2),   
                   jacobian = as.vector(jacobian),
                   XVYXVX = XVYXVX2)    
  }
  
  if('aniso1' %in% paramToEstimate & ('aniso2' %in% paramToEstimate)){
    Output <- list(LogLik=LogLikcpu,
                   breaks = breaks,
                   breaks2d = breaks2d,
                   mleIndex = index,
                   summary = Table,
                   #BetahatTable = x,
                   profsumLogRange = profsumLogRange,
                   profShapeLog = profShapeLog,
                   profNugget = profNugget,
                   profaniso1 = profaniso1,
                   profaniso2 = profaniso2,
                   params = paramsRenew,
                   Infindex = selected_rows,
                   boxcox = boxcox,
                   Nobs = Nobs,
                   Ncov = Ncov,
                   Ndata = Ndata,
                   Nparam = Nparam,
                   ssqY = ssqY2,     
                   ssqBetahat = ssqBetahat2,
                   ssqResidual = ssqResidual2,
                   detVar = as.vector(detVar2),   
                   detReml = as.vector(detReml2),   
                   jacobian = as.vector(jacobian),
                   XVYXVX = XVYXVX2)    
  }  
  Output
  
  
  
}


