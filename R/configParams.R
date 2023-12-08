#' @title set loglikelihood locations
#' @useDynLib gpuLik
#' @export

getHessianNolog <- function(Model,
                            Mle = NULL,
                            boxcox = NULL,
                            data = Model$data, 
                            coordinates = terra::crds(data)
                            ){
  
  delta = 0.01  
  # order the params
  Theorder <- c('range', 'shape', 'nugget', 'anisoRatio', 'anisoAngleRadians')
  parToLog <- c('shape')
  if(is.null(Mle)){
    Mle <- Model$optim$options[,'opt']
  }
  Mle <- Mle[order(match(names(Mle), Theorder))]
  Mle <- Mle[!names(Mle) %in% c('boxcox')]
  
  
  
  if('range' %in% names(Mle)){
    sumLogRange <- log(Mle['range']^2/Mle['anisoRatio'])
  } else{
    sumLogRange = NA
  }
  names(sumLogRange) <- 'sumLogRange'
  
  ## check mle nugget    
  if(('nugget' %in% names(Mle)) & Mle['nugget'] < delta){
    Mle['nugget'] = delta 
  }  
  
  if(!('shape' %in% names(Mle))){
    parToLog <- parToLog[!parToLog %in% 'shape']
  }      
  
  
  if( ('anisoRatio' %in% names(Mle)) & Mle['anisoRatio'] <= 1){
    Mle['anisoRatio'] <- 1/Mle['anisoRatio']
    Mle['anisoAngleRadians'] <- Mle['anisoAngleRadians'] + pi/2 
  }
  
  
  
  whichLogged = which(names(Mle) %in% parToLog)
  whichAniso = which(names(Mle) %in% c('anisoRatio', 'anisoAngleRadians'))
  
  if(('shape' %in% names(Mle)) & Mle['shape'] < 4){
    if('anisoRatio' %in% names(Mle)){
      if(!('anisoAngleRadians' %in% names(Mle))){
        stop('anisoRatio and anisoAngleRadians must be together')
      }
      aniso1 <-  unname(sqrt(Mle['anisoRatio']-1) * cos(2*(Mle['anisoAngleRadians'])))
      aniso2 <-  unname(sqrt(Mle['anisoRatio']-1) * sin(2*(Mle['anisoAngleRadians'])))
      
      aniso <- c(aniso1 = aniso1, aniso2 = aniso2)
      MleGamma = c(sumLogRange, log(Mle[whichLogged]), Mle[-c(1,whichLogged,whichAniso)], aniso)
      
    }else{
      MleGamma = c(sumLogRange, log(Mle[whichLogged]), Mle[-c(1,whichLogged,whichAniso)])
    }
    names(MleGamma)[whichLogged] = paste("log(", names(Mle)[whichLogged], ")",sep="")
    
  }else{ # either no shape or shape > 4
    if('anisoRatio' %in% names(Mle)){
      if(!('anisoAngleRadians' %in% names(Mle))){
        stop('anisoRatio and anisoAngleRadians must be together')
      }
      aniso1 <-  unname(sqrt(Mle['anisoRatio']-1) * cos(2*(Mle['anisoAngleRadians'])))
      aniso2 <-  unname(sqrt(Mle['anisoRatio']-1) * sin(2*(Mle['anisoAngleRadians'])))
      aniso <- c(aniso1 = aniso1, aniso2 = aniso2)
      MleGamma = c(sumLogRange, 1/sqrt(Mle[whichLogged]), Mle[-c(1,whichLogged,whichAniso)], aniso)
      
    }else{
      MleGamma = c(sumLogRange, 1/sqrt(Mle[whichLogged]), Mle[-c(1,whichLogged,whichAniso)])
    }
    names(MleGamma)[whichLogged] = paste("1/sqrt(", names(Mle)[whichLogged], ")",sep="")
  }
  
  
  
  
  
  
  
  # if(Mle['nugget'] > delta){
  ## center approximation
  a1 <- c(1,1,-1,-1)
  a2 <- c(1,-1,1,-1)
  a0 <- rep(0,4)
  if(length(Mle)==1){
    derivGridDf <- matrix(c(-1,0,1), nrow=3, ncol=1)
    
  }else if(length(Mle)==2){
    diagonals <- rbind(rep(0,4),
                       c(1,0,0,0),
                       c(-1,0,0,0),
                       c(0, 1,0,0),
                       c(0,-1,0,0))
    
    derivGridDf <- rbind(
      diagonals,
      cbind(a1, a2, a0, a0))[,1:2]
    
  }else if(length(Mle)==3){
    diagonals <- rbind(rep(0,3),
                       c(1,0,0),
                       c(-1,0,0),
                       c(0,1,0),
                       c(0,-1,0),
                       c(0,0,1),
                       c(0,0,-1))
    derivGridDf <- rbind(
      diagonals,
      cbind(a1, a2, a0),
      cbind(a1, a0, a2),
      cbind(a0, a1, a2))
    
    
  }else if(length(Mle)==4){
    diagonals <- rbind(rep(0,4),
                       c(1,0,0,0),
                       c(-1,0,0,0),
                       c(0,1,0,0),
                       c(0,-1,0,0),
                       c(0,0,1,0),
                       c(0,0,-1,0),
                       c(0,0,0,1),
                       c(0,0,0,-1))
    derivGridDf <- rbind(
      diagonals,
      cbind(a1, a2, a0, a0),
      cbind(a1, a0, a2, a0),
      cbind(a1, a0, a0, a2),
      cbind(a0, a1, a2, a0),
      cbind(a0, a1, a0, a2),
      cbind(a0, a0, a1, a2))
  }else if(length(Mle)==5){
    diagonals <- rbind(rep(0,5),
                       c(1,0,0,0,0),
                       c(-1,0,0,0,0),
                       c(0,1,0,0,0),
                       c(0,-1,0,0,0),
                       c(0,0,1,0,0),
                       c(0,0,-1,0,0),
                       c(0,0,0,1,0),
                       c(0,0,0,-1,0),
                       c(0,0,0,0,1),
                       c(0,0,0,0,-1))
    
    derivGridDf <- rbind(
      diagonals,
      cbind(a1, a2, a0, a0, a0),
      cbind(a1, a0, a2, a0, a0),
      cbind(a1, a0, a0, a2, a0),
      cbind(a1, a0, a0, a0, a2),
      cbind(a0, a1, a2, a0, a0),
      cbind(a0, a1, a0, a2, a0),
      cbind(a0, a1, a0, a0, a2),
      cbind(a0, a0, a1, a2, a0),
      cbind(a0, a0, a1, a0, a2),
      cbind(a0, a0, a0, a1, a2))
  }
  
  
  
  deltas = rep(0.01, length(Mle))
  # names(deltas) = names(MleGamma)
  # deltas[c('log(shape)','log(nugget)')] = 1e-3
  if(length(Mle)<=1){
    ParamsetGamma <- matrix(MleGamma, nrow=nrow(derivGridDf), ncol=length(Mle), byrow=TRUE, dimnames = list(NULL, names(MleGamma))) + derivGridDf*deltas
  }else{
    ParamsetGamma <- matrix(MleGamma, nrow=nrow(derivGridDf), ncol=length(Mle), byrow=TRUE, dimnames = list(NULL, names(MleGamma))) + derivGridDf %*% diag(deltas)
  }
  
  
  temp <- as.data.frame(ParamsetGamma[,'aniso1'] + 1i * ParamsetGamma[,'aniso2'])
  if(('shape' %in% names(Mle)) & Mle['shape'] < 4){
    
    naturalspace <- cbind(Mod(temp[,1])^2 + 1, Arg(temp[,1])/2)
    Paramset <- cbind(sqrt(exp(ParamsetGamma[,'sumLogRange'])*naturalspace[,1]), exp(ParamsetGamma[, whichLogged]), ParamsetGamma[,-c(1,whichLogged,whichAniso)], naturalspace)
    
  }else{
    naturalspace <- cbind(Mod(temp[,1])^2 + 1, Arg(temp[,1])/2)
    Paramset <- cbind(sqrt(exp(ParamsetGamma[,'sumLogRange'])*naturalspace[,1]), (1/ParamsetGamma[, whichLogged])^2, ParamsetGamma[,-c(1,whichLogged,whichAniso)], naturalspace)
  }
  
  colnames(Paramset) <- names(Mle)  
  toAdd = setdiff(c('range','shape','nugget','anisoRatio', 'anisoAngleRadians'), names(Mle))
  otherParams = matrix(Model$opt$mle[toAdd], nrow=nrow(Paramset), ncol = length(toAdd),
                       dimnames = list(rownames(Paramset), toAdd), byrow=TRUE)
  Params1 <- cbind(Paramset, otherParams)
  
  if(is.null(boxcox)){
    boxcox = c(Model$parameters['boxcox']-0.05, Model$parameters['boxcox'], Model$parameters['boxcox']+0.05)
  }
  
  
  
  result1<-gpuLik::getProfLogL(data= data,
                               formula=Model$model$formula,
                               coordinates=coordinates,
                               params=Params1,
                               boxcox = boxcox,
                               type = "double",
                               NparamPerIter=100,
                               gpuElementsOnly=FALSE,
                               reml=Model$model$reml,
                               Nglobal=c(64,64),
                               Nlocal=c(16,8),
                               NlocalCache=2000,
                               verbose=c(0,0))
  
  
  #result$paramsRenew
  # if(length(result1$boxcox)==4){
  #   A <- result1$LogLik[-1, 2]
  #   Origin <- result1$LogLik[1, 2]
  #   boxcoxHessian <- unname((result1$LogLik[1, 4] - 2*result1$LogLik[1, 2] + result1$LogLik[1, 3])/(0.05^2))
  # }else if(length(result1$boxcox)==3){
  A <- result1$LogLik[-1, 1]
  Origin <- result1$LogLik[1, 1]  
  


  boxcoxMle = which.max(result1$LogLik[1,])

  if(boxcoxMle == 1 | boxcoxMle == ncol(result1$LogLik)) warning("boxcox MLE is first or last element")
  boxcoxHessian <- unname(
    (result1$LogLik[1, boxcoxMle-1] - 2*result1$LogLik[1, boxcoxMle] + result1$LogLik[1, boxcoxMle+1]
     )/(diff(boxcox[c(boxcoxMle, boxcoxMle+1)])^2))


  HessianMat <- matrix(0, nrow=length(Mle), ncol=length(Mle))
  rownames(HessianMat) <- names(MleGamma)
  colnames(HessianMat) <- names(MleGamma)
  
  # if(Mle['nugget'] > delta){
  if(length(Mle)==1){
    HessianMat[1,1] <- (result1$LogLik[1, 1] - 2*result1$LogLik[2, 1] + result1$LogLik[3, 1])/(deltas[1]^2)
  }else if(length(Mle)==2){
    HessianMat[1,1] <- (A[1] + A[2] -2*Origin)/(deltas[1]^2)
    HessianMat[2,2] <- (A[3] + A[4] -2*Origin)/(deltas[2]^2)
    HessianMat[1,2] <- (A[5] - A[6] - A[7] + A[8])/(4*deltas[1]*deltas[2])
    HessianMat[2,1] <- HessianMat[1,2]
  }else if(length(Mle)==3){
    HessianMat[1,1] <- (A[1] + A[2] -2*Origin)/(deltas[1]^2)
    HessianMat[2,2] <- (A[3] + A[4] -2*Origin)/(deltas[2]^2)
    HessianMat[3,3] <- (A[5] + A[6] -2*Origin)/(deltas[3]^2)
    HessianMat[1,2] <- (A[7] - A[8] - A[9] + A[10])/(4*deltas[1]*deltas[2])
    HessianMat[1,3] <- (A[11] - A[12] - A[13] + A[14])/(4*deltas[1]*deltas[3])
    HessianMat[2,3] <- (A[15] - A[16] - A[17] + A[18])/(4*deltas[2]*deltas[3])
    HessianMat[2,1] <- HessianMat[1,2]
    HessianMat[3,1] <- HessianMat[1,3]
    HessianMat[3,2] <- HessianMat[2,3]
  }else if(length(Mle)==4){
    HessianMat[1,1] <- (A[1] + A[2] -2*Origin)/(deltas[1]^2)
    HessianMat[2,2] <- (A[3] + A[4] -2*Origin)/(deltas[2]^2)
    HessianMat[3,3] <- (A[5] + A[6] -2*Origin)/(deltas[3]^2)
    HessianMat[4,4] <- (A[7] + A[8] -2*Origin)/(deltas[4]^2)
    
    HessianMat[1,2] <- (A[9] - A[10] - A[11] + A[12])/(4*deltas[1]*deltas[2])
    HessianMat[1,3] <- (A[13] - A[14] - A[15] + A[16])/(4*deltas[1]*deltas[3])
    HessianMat[1,4] <- (A[17] - A[18] - A[19] + A[20])/(4*deltas[1]*deltas[4])
    HessianMat[2,3] <- (A[21] - A[22] - A[23] + A[24])/(4*deltas[2]*deltas[3])
    HessianMat[2,4] <- (A[25] - A[26] - A[27] + A[28])/(4*deltas[2]*deltas[4])
    HessianMat[3,4] <- (A[29] - A[30] - A[31] + A[32])/(4*deltas[3]*deltas[4])
    
  }else if(length(Mle)==5){
    HessianMat[1,1] <- (A[1] + A[2] -2*Origin)/(deltas[1]^2)
    HessianMat[2,2] <- (A[3] + A[4] -2*Origin)/(deltas[2]^2)
    HessianMat[3,3] <- (A[5] + A[6] -2*Origin)/(deltas[3]^2)
    HessianMat[4,4] <- (A[7] + A[8] -2*Origin)/(deltas[4]^2)
    HessianMat[5,5] <- (A[9] + A[10] -2*Origin)/(deltas[5]^2)
    
    HessianMat[1,2] <- (A[11] - A[12] - A[13] + A[14])/(4*deltas[1]*deltas[2])
    HessianMat[1,3] <- (A[15] - A[16] - A[17] + A[18])/(4*deltas[1]*deltas[3])
    HessianMat[1,4] <- (A[19] - A[20] - A[21] + A[22])/(4*deltas[1]*deltas[4])
    HessianMat[1,5] <- (A[23] - A[24] - A[25] + A[26])/(4*deltas[1]*deltas[5])
    HessianMat[2,3] <- (A[27] - A[28] - A[29] + A[30])/(4*deltas[2]*deltas[3])
    HessianMat[2,4] <- (A[31] - A[32] - A[33] + A[34])/(4*deltas[2]*deltas[4])
    HessianMat[2,5] <- (A[35] - A[36] - A[37] + A[38])/(4*deltas[2]*deltas[5])
    HessianMat[3,4] <- (A[39] - A[40] - A[41] + A[42])/(4*deltas[3]*deltas[4])
    HessianMat[3,5] <- (A[43] - A[44] - A[45] + A[46])/(4*deltas[3]*deltas[5])
    HessianMat[4,5] <- (A[47] - A[48] - A[49] + A[50])/(4*deltas[4]*deltas[5])
    
    HessianMat[5,1] <- HessianMat[1,5]
    HessianMat[5,2] <- HessianMat[2,5]
    HessianMat[5,3] <- HessianMat[3,5]
    HessianMat[5,4] <- HessianMat[4,5]
  }
  
  if(length(Mle)==4 | length(Mle)==5){
    HessianMat[2,1] <- HessianMat[1,2]
    HessianMat[3,1] <- HessianMat[1,3]
    HessianMat[3,2] <- HessianMat[2,3]
    HessianMat[4,1] <- HessianMat[1,4]
    HessianMat[4,2] <- HessianMat[2,4]
    HessianMat[4,3] <- HessianMat[3,4]
  }
  
  
  
  boxcoxSetup <- stats::qnorm(
    c(0.01, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.99), 
    mean=Model$parameters['boxcox'], 
    sd = sqrt(abs(1/boxcoxHessian)))

  
  output = list(HessianMat = HessianMat,
                originalPoint = Mle,
                boxcoxSetup = boxcoxSetup,
                fixedVar = toAdd, 
                centralPointGamma = MleGamma,
                parToLog = parToLog,
                whichAniso = whichAniso,
                data = cbind(Params1, result1$LogLik))  

  
  output
}







#' @title set loglikelihood locations
#' @useDynLib gpuLik
#' @export

configParamsSingle <- function(Model,
                         alpha=c(0.001, 0.01, 0.1, 0.2, 0.5, 0.8, 0.95, 0.99),
                         Mle = NULL,
                         boxcox = NULL,
                         shapeRestrict = 1000,
                         data = Model$data, 
                         coordinates = terra::crds(data)
){
  
  
  
  ## get the Hessian

    output <- getHessianNolog(Model = Model,
                              Mle = Mle, 
                              boxcox = boxcox,
                              data = data,
                              coordinates = coordinates) 

  
  Mle <- output$originalPoint
  MleGamma <- output$centralPointGamma
  whichAniso <- output$whichAniso
  parToLog <- output$parToLog
  HessianMat <- output$HessianMat
  boxcoxSetup <- output$boxcoxSetup
  fixedVar <- output$fixedVar
  
  whichLogged = which(names(Mle) %in% parToLog)
  
  
  eigenH = eigen(-HessianMat)
  if(any(eigenH$values>100)){
    eigenH$values = pmax(0.1, eigenH$values) 
  }else{
    eigenH$values = abs(eigenH$values)
  }
  #eigenH$values = pmax(0.01, eigenH$values)
  
  eig = list(values=1/eigenH$values, vectors=eigenH$vectors)
  
  #eig$vectors %*% diag(eig$values) %*% t(eig$vectors)
  
  out_list <- list()
  ## fix first derivative ends  
  #eig <- eigen(Sigma)
  
  # for(i in 1:length(Mle)){
  #   if(eig$values[i] <= 0)
  #     eig$values[i] <- pmin(0.01, abs(eig$values[i]))
  # }
  #eig$values <- abs(eig$values)
  paramsColnames <- c("range","shape","nugget","anisoRatio","anisoAngleRadians")
  
 

  
  if(('shape' %in% names(Mle)) & Mle['shape'] < 4){
    if(length(Mle)==4){
      #load('/home/ruoyong/gpuLik/data/coords4d.RData')
      #system.file("data","coords4d.RData", package = "gpuLik")
      coords4d <- gpuLik:::coords4d
      for(i in 1:length(alpha)){
        clevel <- stats::qchisq(1 - alpha[i], df = 4)
        pointsEllipseGammaspace = t(sqrt(clevel) * eig$vectors %*% diag(sqrt(eig$values)) %*%  t(coords4d) + MleGamma)
        colnames(pointsEllipseGammaspace) <- names(MleGamma)
        temp <- as.data.frame(pointsEllipseGammaspace[,'aniso1'] + 1i * pointsEllipseGammaspace[,'aniso2'])
        naturalspace <- cbind(Mod(temp[,1])^2 + 1, Arg(temp[,1])/2)
        fixed = matrix(Model$opt$mle[fixedVar], nrow=nrow(pointsEllipseGammaspace), ncol = length(fixedVar),
                       dimnames = list(rownames(pointsEllipseGammaspace), fixedVar), byrow=TRUE)
        pointsEllipse <- cbind(sqrt(exp(pointsEllipseGammaspace[,'sumLogRange'])*naturalspace[,1]), exp(pointsEllipseGammaspace[,whichLogged]),pointsEllipseGammaspace[,-c(1,whichLogged,whichAniso)], naturalspace,fixed)
        colnames(pointsEllipse)[1:4] <- names(Mle)
        out_list[[i]] = pointsEllipse[,paramsColnames]
      }
    }else if(length(Mle)==5){
      #load('/home/ruoyong/gpuLik/data/coords5d.RData')
      #system.file("data","coords5d.RData", package = "gpuLik")
      coords5d <- gpuLik:::coords5d
      for(i in 1:length(alpha)){
        clevel <- stats::qchisq(1 - alpha[i], df = 5)
        pointsEllipseGammaspace = t(sqrt(clevel) * eig$vectors %*% diag(sqrt(eig$values)) %*%  t(coords5d) + MleGamma)
        colnames(pointsEllipseGammaspace) <- names(MleGamma)
        temp <- as.data.frame(pointsEllipseGammaspace[,'aniso1'] + 1i * pointsEllipseGammaspace[,'aniso2'])
        naturalspace <- cbind(Mod(temp[,1])^2 + 1, Arg(temp[,1])/2)
        pointsEllipse <- cbind(sqrt(exp(pointsEllipseGammaspace[,'sumLogRange'])*naturalspace[,1]), exp(pointsEllipseGammaspace[,whichLogged]), pointsEllipseGammaspace[,-c(1,whichLogged,whichAniso)], naturalspace)
        colnames(pointsEllipse) <- names(Mle)
        out_list[[i]] = pointsEllipse[,paramsColnames]
      }
    }
  }
  
  
  if(('shape' %in% names(Mle)) & Mle['shape'] >= 4){
    if(length(Mle)==4){
      #load('/home/ruoyong/gpuLik/data/coords4d.RData')
      #system.file("data","coords4d.RData", package = "gpuLik")
      coords4d <- gpuLik:::coords4d
      for(i in 1:length(alpha)){
        clevel <- stats::qchisq(1 - alpha[i], df = 4)
        pointsEllipseGammaspace = t(sqrt(clevel) * eig$vectors %*% diag(sqrt(eig$values)) %*%  t(coords4d) + MleGamma)
        colnames(pointsEllipseGammaspace) <- names(MleGamma)
        temp <- as.data.frame(pointsEllipseGammaspace[,'aniso1'] + 1i * pointsEllipseGammaspace[,'aniso2'])
        naturalspace <- cbind(Mod(temp[,1])^2 + 1, Arg(temp[,1])/2)
        fixed = matrix(Model$opt$mle[fixedVar], nrow=nrow(pointsEllipseGammaspace), ncol = length(fixedVar),
                       dimnames = list(rownames(pointsEllipseGammaspace), fixedVar), byrow=TRUE)
        pointsEllipse <- cbind(sqrt(exp(pointsEllipseGammaspace[,'sumLogRange'])*naturalspace[,1]), (1/pointsEllipseGammaspace[, whichLogged])^2,pointsEllipseGammaspace[,-c(1,whichLogged,whichAniso)], naturalspace, fixed)
        colnames(pointsEllipse)[1:4] <- names(Mle)
        out_list[[i]] = pointsEllipse[,paramsColnames]
      }
    }else if(length(Mle)==5){
      #load('/home/ruoyong/gpuLik/data/coords5d.RData')
      #system.file("data","coords5d.RData", package = "gpuLik")
      coords5d <- gpuLik:::coords5d
      for(i in 1:length(alpha)){
        clevel <- stats::qchisq(1 - alpha[i], df = 5)
        pointsEllipseGammaspace = t(sqrt(clevel) * eig$vectors %*% diag(sqrt(eig$values)) %*%  t(coords5d) + MleGamma)
        colnames(pointsEllipseGammaspace) <- names(MleGamma)
        temp <- as.data.frame(pointsEllipseGammaspace[,'aniso1'] + 1i * pointsEllipseGammaspace[,'aniso2'])
        naturalspace <- cbind(Mod(temp[,1])^2 + 1, Arg(temp[,1])/2)
        pointsEllipse <- cbind(sqrt(exp(pointsEllipseGammaspace[,'sumLogRange'])*naturalspace[,1]), (1/pointsEllipseGammaspace[, whichLogged])^2, pointsEllipseGammaspace[,-c(1,whichLogged,whichAniso)], naturalspace)
        colnames(pointsEllipse) <- names(Mle)
        out_list[[i]] = pointsEllipse[,paramsColnames]
      }
    }
  }
  
  if( !('shape' %in% names(Mle)) & length(Mle)==4){
    #load('/home/ruoyong/gpuLik/data/coords4d.RData')
    #system.file("data","coords4d.RData", package = "gpuLik")
    coords4d <- gpuLik:::coords4d
    for(i in 1:length(alpha)){
      clevel <- stats::qchisq(1 - alpha[i], df = 4)
      pointsEllipseGammaspace = t(sqrt(clevel) * eig$vectors %*% diag(sqrt(eig$values)) %*%  t(coords4d) + MleGamma)
      colnames(pointsEllipseGammaspace) <- names(MleGamma)
      temp <- as.data.frame(pointsEllipseGammaspace[,'aniso1'] + 1i * pointsEllipseGammaspace[,'aniso2'])
      naturalspace <- cbind(Mod(temp[,1])^2 + 1, Arg(temp[,1])/2)
      fixed = matrix(Model$opt$mle[fixedVar], nrow=nrow(pointsEllipseGammaspace), ncol = length(fixedVar),
                     dimnames = list(rownames(pointsEllipseGammaspace), fixedVar), byrow=TRUE)
      pointsEllipse <- cbind(sqrt(exp(pointsEllipseGammaspace[,'sumLogRange'])*naturalspace[,1]), exp(pointsEllipseGammaspace[,whichLogged]),pointsEllipseGammaspace[,'nugget'], naturalspace, fixed)
      colnames(pointsEllipse)[1:4] <- names(Mle)
      out_list[[i]] = pointsEllipse[,paramsColnames]
    }
  }      
  
  
  
  
  
  
  for(i in 1:length(alpha)){
    vector1 <- out_list[[i]][,'anisoAngleRadians']
    for(j in 1:length(vector1)){
      if(vector1[j] > pi/2){
        vector1[j] <- vector1[j] - pi
        # if (vector1[j] > pi/2)
        #   vector1[j] <- vector1[j] - pi
      }else if (vector1[j] < -pi/2){
        vector1[j] <- vector1[j] + pi
        # if (vector1[j] < -pi/2)
        #   vector1[j] <- vector1[j] + pi
      }
    }
    out_list[[i]][,'anisoAngleRadians'] <- vector1
  }
  
  
  
  
  
  if(('nugget' %in% names(Mle))){
    for(i in 1:length(alpha)){
      vector <- out_list[[i]][,'nugget']
      for(j in 1:(length(vector)*1/2)){
        if(vector[j]<0){
          vector[j] = 0 
        }
      }
      for(j in (length(vector)*1/2+1):length(vector)){
        if(vector[j]<0){
          vector[j] = stats::runif(1, 0, 2) #-vector[j]#
        }
      }
      out_list[[i]][,'nugget'] <- vector
    }
  }
  
  # if('shape' %in% names(Mle) & Mle['shape'] >= 4){
  # for(i in 1:length(alpha)){
  #   vector1 <- 1/out_list[[i]][,'shape']
  #   vector2 <- rep(1000, length(vector))
  #   out_list[[i]][,'shape'] = pmin(vector1, vector2)
  # }
  # }
  if('shape' %in% names(Mle)  ){
    for(i in 1:length(alpha)){
      vector <- out_list[[i]][,'shape']
      for(j in 1:length(vector)){
        if(vector[j]>shapeRestrict)
          vector[j] = stats::runif(1, 0.1, shapeRestrict)
      }
      out_list[[i]][,'shape'] = vector
    }
  }
  
  
  # names(out_list) <- paste0("alpha", alpha, sep="")
  # out_list[[length(alpha)+1]] <- boxcoxSetup
  
  outPut <- list()
  outPut[[1]] = do.call(rbind, out_list[1:length(alpha)])
  
  if(length(Mle)==5){
    repN = 726
    outPut[[1]] <- cbind(outPut[[1]], alpha = rep(alpha, each=repN))
  }else if(length(Mle)==4){
    repN = 120
    outPut[[1]] <- cbind(outPut[[1]], alpha = rep(alpha, each=repN))
  }
  
  # outPut[[1]] = rbind(optParams,   outPut[[1]])
  rownames(outPut[[1]]) <- NULL
  outPut[[2]] <- boxcoxSetup
  names(outPut) <- c('representativeParamaters', 'boxcox')
  outPut
 
}




#' @title set loglikelihood locations
#' @useDynLib gpuLik
#' @export   

configParams = function(Model,  # note that the model which does not fix any parameter should always be put the first in the list!!!
                        alpha=c(0.001, 0.01, 0.1, 0.2, 0.5, 0.8, 0.95, 0.99),
                        alphasecond = NULL,
                        Mle = NULL,
                        boxcox = NULL,# a vector of confidence levels 1-alpha
                        shapeRestrict = 1000,
                        data = Model[[1]]$data,
                        coordinates = terra::crds(data)){
  
   if(is.null(alphasecond)){
     result = lapply(Model, configParamsSingle,
                     alpha = alpha, 
                     Mle = Mle,
                     boxcox = boxcox,
                     shapeRestrict = shapeRestrict, data=data, coordinates=coordinates)
     
     resultInMatrix = list()
     
     for(D in 1:length(result)) {
       resultInMatrix[[D]] = result[[D]]$representativeParamaters
     }  
     
     boxcox = result[[1]]$boxcox
     paramsAll = do.call(rbind, resultInMatrix)
   }else{ # have alphasecond
     
     firstmodel <- configParamsSingle(Model[[1]], 
                                      alpha=alpha,
                                      data=data,
                                      coordinates=coordinates)
     boxcox = firstmodel$boxcox
     restlist = list()
     for (i in 2:length(Model)){
       restlist[i-1] = Model[i]
     }
     

     result = lapply(restlist, configParamsSingle,
                     alpha = alphasecond, 
                     Mle = Mle,
                     boxcox = boxcox,
                     shapeRestrict = shapeRestrict, 
                     data = data, 
                     coordinates = coordinates)
     
     resultInMatrix = list()
     
     for(D in 1:length(result)){
       resultInMatrix[[D]] = result[[D]]$representativeParamaters
     }  
     
     paramsAll = do.call(rbind, resultInMatrix)
     paramsAll = rbind(firstmodel$representativeParamaters,paramsAll)
     }

  paramsColnames = colnames(paramsAll)[1:5]
  
  optParams = lapply(Model, function(xx) xx$opt$mle[paramsColnames])
  optParams = do.call(rbind, optParams)
  optParams = cbind(optParams, alpha = NA)
  
  paramsAll = rbind(optParams,paramsAll)

  
  result = list(
    representativeParamaters = paramsAll,
    boxcox = boxcox,
    reml = Model[[1]]$model$reml,
    alpha = alpha
  )
  
  result    
}    




































