#' @title set loglikelihood locations
#' @useDynLib gpuLik
#' @export

getHessianNolog <- function(Model,
                       Mle = NULL,
                       boxcox = NULL){
  
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

  }else{
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
    boxcox = c(Model$parameters['boxcox'], Model$parameters['boxcox']-0.05, Model$parameters['boxcox']+0.05)
  }
  
  
  
  result1<-gpuLik::getProfLogL(data= Model$data,
                                  formula=Model$model$formula,
                                  coordinates=Model$data@coords,
                                  params=Params1,
                                  boxcox = boxcox,
                                  type = "double",
                                  NparamPerIter=100,
                                  gpuElementsOnly=FALSE,
                                  reml=FALSE,
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
    boxcoxHessian <- unname((result1$LogLik[1, 3] - 2*result1$LogLik[1, 1] + result1$LogLik[1, 2])/(0.05^2))
  # }
  #plot(result$LogLik[])
  
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
  
  

  boxcoxSetup <- stats::qnorm(c(0.01, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.99), mean=Model$parameters['boxcox'], sd = sqrt(solve((-boxcoxHessian))))
  
  


  # frst detivatives
  # if((('nugget' %in% names(Mle)) & Mle['nugget'] < delta) | ('shape' %in% names(Mle)) & Mle['shape'] >= 4){
  #   derivGridDf1 <- rbind(c(1,0,0,0,0),
  #                         c(-1, 0,0,0,0),
  #                         c(0, 1, 0,0,0),
  #                         c(0,-1, 0,0,0),
  #                         c(0, 0, 1,0,0),
  #                         c(0, 0,-1,0,0),
  #                         c(0, 0, 0,1,0),
  #                         c(0, 0, 0,-1,0),
  #                         c(0, 0, 0, 0,1),
  #                         c(0, 0, 0, 0,-1))
  #   if(length(Mle)==4){
  #     derivGridDf1 <- derivGridDf1[-c(9,10), -5]
  #   }else if(length(Mle)==2){
  #     derivGridDf1 <- derivGridDf1[c(1:4), 1:2]
  #   }else if(length(Mle) == 3){
  #     derivGridDf1 <- derivGridDf1[1:6, 1:3]
  #   }
  # 
  # 
  #   deltas = rep(delta, length(Mle))
  #   # names(deltas) = names(MleGamma)
  #   # deltas['aniso2'] = 0.02
  #   ParamsetGamma <- matrix(MleGamma, nrow=nrow(derivGridDf1), ncol=length(Mle), byrow=TRUE, dimnames = list(NULL, names(MleGamma))) +
  #     derivGridDf1 %*% diag(deltas)
  # 
  #   # if(!('anisoRatio' %in% names(Mle))){
  #   #   Paramset <- cbind(ParamsetGamma[,-whichLogged], exp(ParamsetGamma[,paste("log(", names(Mle)[whichLogged], ")",sep="")]))
  #   # }else{
  #   temp <- as.data.frame(ParamsetGamma[,'aniso1'] + 1i * ParamsetGamma[,'aniso2'])
  #     if(Mle['anisoRatio'] <= 1){
  #       naturalspace <- cbind(1/(Mod(temp[,1])^2 + 1), Arg(temp[,1])/2 + pi/2)
  #       Paramset <- cbind(sqrt(exp(ParamsetGamma[,'sumLogRange'])*naturalspace[,1]), (1/ParamsetGamma[, whichLogged])^2, ParamsetGamma[,'nugget'], naturalspace)
  #     }else{
  #       naturalspace <- cbind(Mod(temp[,1])^2 + 1, Arg(temp[,1])/2)
  #       Paramset <- cbind(sqrt(exp(ParamsetGamma[,'sumLogRange'])*naturalspace[,1]), (1/ParamsetGamma[, whichLogged])^2, ParamsetGamma[,'nugget'], naturalspace)
  #     }
  #   # }
  # 
  #   colnames(Paramset) <- names(Mle)
  #   toAdd = setdiff(c('range','shape','nugget','anisoRatio', 'anisoAngleRadians'), names(Mle))
  #   otherParams = matrix(Model$opt$mle[toAdd], nrow=nrow(Paramset), ncol = length(toAdd),
  #                        dimnames = list(rownames(Paramset), toAdd), byrow=TRUE)
  #   Params2 <- cbind(Paramset, otherParams)
  # 
  #   result2<-gpuLik::getProfLogL(data= Model$data,
  #                                   formula=Model$model$formula,
  #                                   coordinates=Model$data@coords,
  #                                   params=Params2,
  #                                   boxcox = Model$parameters['boxcox'],
  #                                   type = "double",
  #                                   NparamPerIter=20,
  #                                   gpuElementsOnly=FALSE,
  #                                   reml=FALSE,
  #                                   Nglobal=c(128,64),
  #                                   Nlocal=c(16,16),
  #                                   NlocalCache=2000)
  # 
  #   if(length(result1$boxcox)==2){
  #     A <- result2$LogLik[, 2]
  #   }else{
  #     A <- result2$LogLik[, 1]
  #   }
  #   FirstDeri <- rep(0, length(Mle))
  #   names(FirstDeri) <- names(MleGamma)
  # 
  #   FirstDeri[1] <- (A[1] - A[2])/(2*deltas[1])
  #   FirstDeri[2] <- (A[3] - A[4])/(2*deltas[2])
  #   FirstDeri[3] <- (A[5] - A[6])/(2*deltas[3])
  #   FirstDeri[4] <- (A[7] - A[8])/(2*deltas[4])
  #   FirstDeri[5] <- (A[9] - A[10])/(2*deltas[5])
  # 
  #   index <- which(abs(FirstDeri) > 0.01)
  # }
  # 
  # #Sigma <- -solve(HessianMat)
  # 
  # newMle <- Mle
  # newMleGamma <- MleGamma
  # 
  # 
  # if((('nugget' %in% names(Mle)) & Mle['nugget'] < delta) | ('shape' %in% names(Mle)) & Mle['shape'] >= 4){
  # 
  #   for(i in 1:length(index)){
  #     #newMle[index[i]] <- Mle[index[i]] + Sigma[index[i],index[i]]*FirstDeri[index[i]]
  #     newMleGamma[index[i]] <- MleGamma[index[i]] + solve(-HessianMat)[index[i],index[i]]*FirstDeri[index[i]]
  #   }
  # 
  # 
  #     temp <- as.data.frame(newMleGamma['aniso1'] + 1i * newMleGamma['aniso2'])
  #     if(Mle['anisoRatio'] <= 1){
  #       naturalspace <- c(1/(Mod(temp[,1])^2 + 1), Arg(temp[,1])/2 + pi/2)
  #       newMle <- c(sqrt(exp(newMleGamma['sumLogRange'])*naturalspace[1]), (1/newMleGamma[ whichLogged])^2, newMleGamma['nugget'], naturalspace)
  #     }else{
  #       naturalspace <- c(Mod(temp[,1])^2 + 1, Arg(temp[,1])/2 )
  #       newMle <- c(sqrt(exp(newMleGamma['sumLogRange'])*naturalspace[1]), (1/newMleGamma[ whichLogged])^2, newMleGamma['nugget'], naturalspace)
  #     }
  # 
  #   names(newMle) <- names(Mle)
  # 
  # 
  #   # check mle shape
  #   # if('shape' %in% names(Mle) & newMle['shape'] >= kappa){
  #   #   newMle['shape'] = 1/newMle['shape']
  #   # }
  # }
  
  
  # if((('nugget' %in% names(Mle)) & Mle['nugget'] < delta) | ('shape' %in% names(Mle)) & Mle['shape'] >= kappa){
  #   output = list(Gradiant= FirstDeri,
  #                 HessianMat = HessianMat,
  #                 originalPoint = Mle,
  #                 centralPoint = newMle,
  #                 centralPointGamma = newMleGamma,
  #                 parToLog = parToLog,
  #                 whichAniso = whichAniso,
  #                 data = cbind(Params1, result1$LogLik))
  # }else{
    output = list(HessianMat = HessianMat,
                  originalPoint = Mle,
                  boxcoxSetup = boxcoxSetup,
                  fixedVar = toAdd, 
                  centralPointGamma = MleGamma,
                  parToLog = parToLog,
                  whichAniso = whichAniso,
                  data = cbind(Params1, result1$LogLik))  
  # }
  
  output
}



















    getHessianLog <- function(Model,
                           Mle = NULL,
                           boxcox = NULL){
  
      delta = 0.01  
      kappa = 60
      # order the params
      Theorder <- c('range', 'shape', 'nugget', 'anisoRatio', 'anisoAngleRadians')
      parToLog <- c('shape', "nugget")
      if(is.null(Mle)){
        Mle <- Model$optim$options[,'opt']
      }
      Mle <- Mle[order(match(names(Mle), Theorder))]
      Mle <- Mle[!names(Mle) %in% c('boxcox')]
      #Mle <- Mle[!names(Mle) %in% c('shape')]
      
      if('range' %in% names(Mle)){
        sumLogRange <- log(Mle['range']^2/Mle['anisoRatio'])
      }
      names(sumLogRange) <- 'sumLogRange'
    
      ## check mle nugget    
      if(('nugget' %in% names(Mle)) & Mle['nugget'] < delta){
        Mle['nugget'] = delta 
        parToLog <- parToLog[!parToLog %in% 'nugget']
      }  
     
      if(!('nugget' %in% names(Mle))){
        parToLog <- parToLog[!parToLog %in% 'nugget']
      }
      # if(!('shape' %in% names(Mle))){
      #   parToLog <- parToLog[!parToLog %in% 'shape']
      # }      
      
      
      whichLogged = which(names(Mle) %in% parToLog)
      whichAniso = which(names(Mle) %in% c('anisoRatio', 'anisoAngleRadians'))
      
      if(('shape' %in% names(Mle)) & Mle['shape'] < 4){
        if('anisoRatio' %in% names(Mle)){
          if(!('anisoAngleRadians' %in% names(Mle))){
            stop('anisoRatio and anisoAngleRadians must be together')
          }
          if(Mle['anisoRatio'] <= 1){
            aniso1 <-  unname(sqrt(1/Mle['anisoRatio']-1) * cos(2*(Mle['anisoAngleRadians'] + pi/2)))
            aniso2 <-  unname(sqrt(1/Mle['anisoRatio']-1) * sin(2*(Mle['anisoAngleRadians'] + pi/2)))
          }else{
            aniso1 <-  unname(sqrt(Mle['anisoRatio']-1) * cos(2*(Mle['anisoAngleRadians'])))
            aniso2 <-  unname(sqrt(Mle['anisoRatio']-1) * sin(2*(Mle['anisoAngleRadians'])))
          }
          aniso <- c(aniso1 = aniso1, aniso2 = aniso2)
          MleGamma = c(sumLogRange, log(Mle[whichLogged]), Mle[-c(1,whichLogged,whichAniso)], aniso)
          
        }else{
          MleGamma = c(sumLogRange, log(Mle[whichLogged]), Mle[-c(1,whichLogged,whichAniso)])
        }
        names(MleGamma)[whichLogged] = paste("log(", names(Mle)[whichLogged], ")",sep="")
        
      }else{
        if('anisoRatio' %in% names(Mle)){
          if(!('anisoAngleRadians' %in% names(Mle))){
            stop('anisoRatio and anisoAngleRadians must be together')
          }
          if(Mle['anisoRatio'] <= 1){
            aniso1 <-  unname(sqrt(1/Mle['anisoRatio']-1) * cos(2*(Mle['anisoAngleRadians'] + pi/2)))
            aniso2 <-  unname(sqrt(1/Mle['anisoRatio']-1) * sin(2*(Mle['anisoAngleRadians'] + pi/2)))
          }else{
            aniso1 <-  unname(sqrt(Mle['anisoRatio']-1) * cos(2*(Mle['anisoAngleRadians'])))
            aniso2 <-  unname(sqrt(Mle['anisoRatio']-1) * sin(2*(Mle['anisoAngleRadians'])))
          }
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
  

  # }else{
  # ## forward approximation
  # a0 <- rep(0,4)
  # a06 <- rep(0,6)
  # a1 <- c(0,1,2, 0, 1, 2)
  # a2 <- c(1,1,1,-1,-1,-1)
  # a3 <- c(1,1,-1,-1)
  # a4 <- c(1,-1,1,-1)
  # if(length(Mle)==2){
  #   diagonals <- rbind(rep(0,4),
  #                      c(1,0,0,0),
  #                      c(-1,0,0,0),
  #                      c(0,1,0,0),
  #                      c(0,2,0,0),
  #                      c(0,3,0,0))
  #   derivGridDf <- rbind(
  #     diagonals,
  #     cbind(a2, a1, a06, a06))[,1:2]
  #   
  # }else if(length(Mle)==4){
  #   diagonals <- rbind(rep(0,4),
  #                      c(1,0,0,0),
  #                      c(-1,0,0,0),
  #                      c(0,1,0,0),
  #                      c(0,2,0,0),
  #                      c(0,3,0,0),
  #                      c(0,0,1,0),
  #                      c(0,0,-1,0),
  #                      c(0,0,0,1),
  #                      c(0,0,0,-1))
  #   derivGridDf <- rbind(
  #     diagonals,
  #     cbind(a2, a1, a06, a06),
  #     cbind(a3, a0, a4, a0),
  #     cbind(a3, a0, a0, a4),
  #     cbind(a06, a1, a2, a06),
  #     cbind(a06, a1, a06, a2),
  #     cbind(a0, a0, a3, a4))
  # }else if(length(Mle)==5){
  #   diagonals <- rbind(rep(0,5),
  #                      c(1,0,0,0,0),
  #                      c(-1,0,0,0,0),
  #                      c(0,1,0, 0,0),
  #                      c(0,-1,0, 0,0),
  #                      c(0,0,1,0,0),
  #                      c(0,0,2,0,0),
  #                      c(0,0,3,0,0),
  #                      c(0,0,0,1,0),
  #                      c(0,0,0,-1,0),
  #                      c(0,0,0,0,1),
  #                      c(0,0,0,0,-1))
  #   derivGridDf <- rbind(
  #     diagonals,
  #     cbind(a3, a4, a0, a0, a0),
  #     cbind(a2, a06, a1, a06, a06),
  #     cbind(a3, a0, a0, a4, a0),
  #     cbind(a3, a0, a0, a0, a4),
  #     cbind(a06, a2, a1, a06, a06),
  #     cbind(a0, a3, a0, a4, a0),
  #     cbind(a0, a3, a0, a0, a4),
  #     cbind(a06, a06, a1, a2, a06),
  #     cbind(a06, a06, a1, a06, a2),
  #     cbind(a0, a0, a0, a3, a4))
  # }
  # }
  
 #  delta = 0.5
 #  a <- list(sumLogRange=seq(-3*delta, 3*delta, by=delta),
 #            logshape=seq(-3*delta, 3*delta, by=delta),
 #            lognugget=seq(-3*delta, 3*delta, by=delta),
 #            aniso1=seq(-3*delta, 3*delta, by=delta),
 #            aniso2=seq(-3*delta, 3*delta, by=delta))
 #  
 #  aa <- do.call(expand.grid, a)
 #  nrow(aa)
 #  PGCheck <- matrix(MleGamma, nrow=nrow(aa), ncol=length(Mle), byrow=TRUE, dimnames = list(NULL, names(MleGamma))) + aa 
 #  temp <- as.data.frame(PGCheck[,'aniso1'] + 1i * PGCheck[,'aniso2'])
 #  naturalspace <- cbind(Mod(temp[,1])^2 + 1, Arg(temp[,1])/2+pi/2)
 #  ParamsetCheck <- cbind(sqrt(exp(PGCheck[,'sumLogRange'])*naturalspace[,1]), exp(PGCheck[, whichLogged]), naturalspace)
 #  colnames(ParamsetCheck) <- names(Mle)  
 #  toAdd = setdiff(c('range','shape','nugget','anisoRatio', 'anisoAngleRadians'), names(Mle))
 #  otherParams = matrix(Model$opt$mle[toAdd], nrow=nrow(ParamsetCheck), ncol = length(toAdd),
 #                       dimnames = list(rownames(ParamsetCheck), toAdd), byrow=TRUE)
 #  ParamsCheck1 <- cbind(ParamsetCheck, otherParams)
 #  resultC<-gpuLik::getProfLogL(data= Model$data,
 #                                  formula=Model$model$formula,
 #                                  coordinates=Model$data@coords,
 #                                  params= ParamsCheck1,
 #                                  boxcox = Model$parameters['boxcox'],
 #                                  type = "double",
 #                                  NparamPerIter=200,
 #                                  gpuElementsOnly=FALSE,
 #                                  reml=FALSE,
 #                                  Nglobal=c(128,64),
 #                                  Nlocal=c(16,16),
 #                                  NlocalCache=2000)
 #  
 # which(resultC$LogLik==max(resultC$LogLik), arr.ind = TRUE)
 # resultC$LogLik[16569,]
 # ParamsCheck1[16569,] 
 # ParamsCheck1[8404,]
 # for(i in 1:nrow(aa)){
 #   if (all(aa[i,] == c(0,0,0,0,0)))
 #     print(i)
 # }

 # CList <- c(range=29654.39, shape=0.4291865, nugget=0.3636058, anisoRatio=1.870348, anisoAngleRadians=0.9481546, boxcox=1.070506)
 # CList <- c(range=29654.39, shape=0.4291865, nugget=0.3636058, anisoRatio=1.870348, anisoAngleRadians=0.9481546, boxcox=1.070506)
 # a <- geostatsp::loglikLgm(
 #   c(CList['range'],
 #     CList['shape'],
 #     CList['nugget'],
 #     CList[c('anisoRatio')],
 #     CList[c('anisoAngleDegrees')],
 #     CList['boxcox']),
 #   data = mydat,
 #   formula = as.formula(paste(colnames(mydat@data)[i+2], "~", "cov1 + cov2")),
 #   reml = FALSE,
 #   minustwotimes=FALSE)[['logLik']]
 
  deltas = rep(0.01, length(Mle))
  # names(deltas) = names(MleGamma)
  # deltas[c('log(shape)','log(nugget)')] = 1e-3
  if(length(Mle)<=1){
    ParamsetGamma <- matrix(MleGamma, nrow=nrow(derivGridDf), ncol=length(Mle), byrow=TRUE, dimnames = list(NULL, names(MleGamma))) + derivGridDf*deltas
  }else{
    ParamsetGamma <- matrix(MleGamma, nrow=nrow(derivGridDf), ncol=length(Mle), byrow=TRUE, dimnames = list(NULL, names(MleGamma))) + derivGridDf %*% diag(deltas)
  }
  
  if(('shape' %in% names(Mle)) & Mle['shape'] < 4){  
  if(!('anisoRatio' %in% names(Mle))){
    Paramset <- cbind(ParamsetGamma[,-whichLogged], exp(ParamsetGamma[,paste("log(", names(Mle)[whichLogged], ")",sep="")]))   
  }else{
    temp <- as.data.frame(ParamsetGamma[,'aniso1'] + 1i * ParamsetGamma[,'aniso2'])
    if(Mle['anisoRatio'] <= 1){
    naturalspace <- cbind(1/(Mod(temp[,1])^2 + 1), Arg(temp[,1])/2 + pi/2)
    Paramset <- cbind(sqrt(exp(ParamsetGamma[,'sumLogRange'])*naturalspace[,1]), exp(ParamsetGamma[, whichLogged]), naturalspace)
    }else{
    naturalspace <- cbind(Mod(temp[,1])^2 + 1, Arg(temp[,1])/2 )
    if(length(whichLogged)>=2 & whichLogged[2]-whichLogged[1]>1){
      Paramset <- cbind(exp(ParamsetGamma[, whichLogged[1]]), ParamsetGamma[,-c(whichLogged, whichAniso)], exp(ParamsetGamma[, whichLogged[2]]), naturalspace)
    }else{
      Paramset <- cbind(sqrt(exp(ParamsetGamma[,'sumLogRange'])*naturalspace[,1]), exp(ParamsetGamma[, whichLogged]), naturalspace)
    }
    }
  }
  }else{
    temp <- as.data.frame(ParamsetGamma[,'aniso1'] + 1i * ParamsetGamma[,'aniso2'])
    if(Mle['anisoRatio'] <= 1){
      naturalspace <- cbind(1/(Mod(temp[,1])^2 + 1), Arg(temp[,1])/2 + pi/2)
      Paramset <- cbind(sqrt(exp(ParamsetGamma[,'sumLogRange'])*naturalspace[,1]), (1/ParamsetGamma[, whichLogged])^2, ParamsetGamma[,-c(1,whichLogged,whichAniso)], naturalspace)
    }else{
      naturalspace <- cbind(Mod(temp[,1])^2 + 1, Arg(temp[,1])/2)
      Paramset <- cbind(sqrt(exp(ParamsetGamma[,'sumLogRange'])*naturalspace[,1]), (1/ParamsetGamma[, whichLogged])^2, ParamsetGamma[,-c(1,whichLogged,whichAniso)], naturalspace)
    }    
  }
  
  colnames(Paramset) <- names(Mle)  
  toAdd = setdiff(c('range','shape','nugget','anisoRatio', 'anisoAngleRadians'), names(Mle))
  otherParams = matrix(Model$opt$mle[toAdd], nrow=nrow(Paramset), ncol = length(toAdd),
                       dimnames = list(rownames(Paramset), toAdd), byrow=TRUE)
  Params1 <- cbind(Paramset, otherParams)
  
  if(is.null(boxcox)){
    boxcox = Model$parameters['boxcox']
  }
  
  
  
  result1<-gpuLik::getProfLogL(data= Model$data,
                                 formula=Model$model$formula,
                                 coordinates=Model$data@coords,
                                 params=Params1,
                                 boxcox = boxcox,
                                 type = "double",
                                 NparamPerIter=100,
                                 gpuElementsOnly=FALSE,
                                 reml=FALSE,
                                 Nglobal=c(128,64),
                                 Nlocal=c(16,8),
                                 NlocalCache=2000)
  
  
  #result$paramsRenew
  # if(length(result1$boxcox)==2){
  # A <- result1$LogLik[-1, 2]
  # Origin <- result1$LogLik[1, 2]
  # }else{
  A <- result1$LogLik[-1, 1]
  Origin <- result1$LogLik[1, 1]  
  # }
  #plot(result$LogLik[])
  
  HessianMat <- matrix(0, nrow=length(Mle), ncol=length(Mle))
  rownames(HessianMat) <- names(MleGamma)
  colnames(HessianMat) <- names(MleGamma)
  
  # if(Mle['nugget'] > delta){
  if(length(Mle)==1){
    HessianMat[1,1] <- (result$LogLik[1, 2] - 2*result$LogLik[2, 2] + result$LogLik[3, 2])/(deltas[1]^2)
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
  # }else{
  #   if(length(Mle)==4){
  #   HessianMat[1,1] <- (A[1] + A[2] -2*Origin)/(0.01^2)
  #   HessianMat[2,2] <- (2*Origin - 5*A[3] + 4*A[4] - A[5])/(0.01^2)
  #   HessianMat[3,3] <- (A[6] + A[7] -2*Origin)/(0.01^2)
  #   HessianMat[4,4] <- (A[8] + A[9] -2*Origin)/(0.01^2)
  #   
  #   HessianMat[1,2] <- (-3*A[10] + 4*A[11] - A[12] + 3*A[13] - 4*A[14] + A[15])/(4*0.01^2)
  #   HessianMat[1,3] <- (A[16] - A[17] - A[18] + A[19])/(4*0.01^2)
  #   HessianMat[1,4] <- (A[20] - A[21] - A[22] + A[23])/(4*0.01^2)
  #   HessianMat[2,3] <- (-3*A[24] + 4*A[25] - A[26] + 3*A[27] - 4*A[28] + A[29])/(4*0.01^2)
  #   HessianMat[2,4] <- (-3*A[30] + 4*A[31] - A[32] + 3*A[33] - 4*A[34] + A[35])/(4*0.01^2)
  #   HessianMat[3,4] <- (A[36] - A[37] - A[38] + A[39])/(4*0.01^2)
  #   
  #   }else if(length(Mle)==5){
  #     HessianMat[1,1] <- (A[1] + A[2] -2*Origin)/(0.01^2)
  #     HessianMat[2,2] <- (A[3] + A[4] -2*Origin)/(0.01^2)
  #     HessianMat[3,3] <- (2*Origin - 5*A[5] + 4*A[6] - A[7])/(0.01^2)
  #     HessianMat[4,4] <- (A[8] + A[9] -2*Origin)/(0.01^2)
  #     HessianMat[5,5] <- (A[10] + A[11] -2*Origin)/(0.01^2)
  #     
  #     HessianMat[1,2] <- (A[12] - A[13] - A[14] + A[15])/(4*0.01^2)
  #     HessianMat[1,3] <- (-3*A[16] + 4*A[17] - A[18] + 3*A[19] - 4*A[20] + A[21])/(4*0.01^2)
  #     HessianMat[1,4] <- (A[22] - A[23] - A[24] + A[25])/(4*0.01^2)
  #     HessianMat[1,5] <- (A[26] - A[27] - A[28] + A[29])/(4*0.01^2)
  #     HessianMat[2,3] <- (-3*A[30] + 4*A[31] - A[32] + 3*A[33] - 4*A[34] + A[35])/(4*0.01^2)
  #     HessianMat[2,4] <- (A[36] - A[37] - A[38] + A[39])/(4*0.01^2)
  #     HessianMat[2,5] <- (A[40] - A[41] - A[42] + A[43])/(4*0.01^2)
  #     HessianMat[3,4] <- (-3*A[44] + 4*A[45] - A[46] + 3*A[47] - 4*A[48] + A[49])/(4*0.01^2)
  #     HessianMat[3,5] <- (-3*A[50] + 4*A[51] - A[52] + 3*A[53] - 4*A[54] + A[55])/(4*0.01^2)
  #     HessianMat[4,5] <- (A[56] - A[57] - A[58] + A[59])/(4*0.01^2)
  #     
  #     HessianMat[5,1] <- HessianMat[1,5]
  #     HessianMat[5,2] <- HessianMat[2,5]
  #     HessianMat[5,3] <- HessianMat[3,5]
  #     HessianMat[5,4] <- HessianMat[4,5]  
  #     
  #     if(length(Mle)==4 | length(Mle)==5){
  #       HessianMat[2,1] <- HessianMat[1,2]
  #       HessianMat[3,1] <- HessianMat[1,3]
  #       HessianMat[3,2] <- HessianMat[2,3]
  #       HessianMat[4,1] <- HessianMat[1,4]
  #       HessianMat[4,2] <- HessianMat[2,4]
  #       HessianMat[4,3] <- HessianMat[3,4]
  #     }      
  #     
  # }
  # }  
  
#   if(length(Mle)==5){
#     HessianMat[,2] <- (0.5) * HessianMat[,2]
#     HessianMat[2,] <- (0.5) * HessianMat[2,]
# 
#   #HessianMat2 <- diag(c(1, 0.5,1,1,1)) %*% HessianMat %*% diag(c(1, 0.5,1,1,1))
# }
  # frst detivatives
  # if((('nugget' %in% names(Mle)) & Mle['nugget'] < delta) | ('shape' %in% names(Mle)) & Mle['shape'] >= kappa){
  # derivGridDf1 <- rbind(c(1,0,0,0,0),
  #                       c(-1, 0,0,0,0),
  #                       c(0, 1, 0,0,0),
  #                       c(0,-1, 0,0,0),
  #                       c(0, 0, 1,0,0),
  #                       c(0, 0,-1,0,0),
  #                       c(0, 0, 0,1,0),
  #                       c(0, 0, 0,-1,0),
  #                       c(0, 0, 0, 0,1),
  #                       c(0, 0, 0, 0,-1))
  # if(length(Mle)==4){
  # derivGridDf1 <- derivGridDf1[-c(9,10), -5]
  # }else if(length(Mle)==2){
  # derivGridDf1 <- derivGridDf1[c(1:4), 1:2]  
  # }else if(length(Mle) == 3){
  # derivGridDf1 <- derivGridDf1[1:6, 1:3]
  # }
  # 
  # 
  # deltas = rep(delta, length(Mle))
  # # names(deltas) = names(MleGamma)
  # # deltas['aniso2'] = 0.02
  # ParamsetGamma <- matrix(MleGamma, nrow=nrow(derivGridDf1), ncol=length(Mle), byrow=TRUE, dimnames = list(NULL, names(MleGamma))) + 
  #   derivGridDf1 %*% diag(deltas)
  # 
  # if(!('anisoRatio' %in% names(Mle))){
  #   Paramset <- cbind(ParamsetGamma[,-whichLogged], exp(ParamsetGamma[,paste("log(", names(Mle)[whichLogged], ")",sep="")]))   
  # }else{
  #   temp <- as.data.frame(ParamsetGamma[,'aniso1'] + 1i * ParamsetGamma[,'aniso2'])
  #   if(Mle['anisoRatio'] <= 1){
  #     naturalspace <- cbind(1/(Mod(temp[,1])^2 + 1), Arg(temp[,1])/2 + pi/2)
  #     Paramset <- cbind(sqrt(exp(ParamsetGamma[,'sumLogRange'])*naturalspace[,1]), exp(ParamsetGamma[, whichLogged]), naturalspace)
  #   }else{
  #     naturalspace <- cbind(Mod(temp[,1])^2 + 1, Arg(temp[,1])/2)
  #     if(length(whichLogged)>=2 & whichLogged[2]-whichLogged[1]>1){
  #       Paramset <- cbind(exp(ParamsetGamma[, whichLogged[1]]), ParamsetGamma[,-c(whichLogged, whichAniso)], exp(ParamsetGamma[, whichLogged[2]]), naturalspace)
  #     }else{
  #       Paramset <- cbind(sqrt(exp(ParamsetGamma[,'sumLogRange'])*naturalspace[,1]), exp(ParamsetGamma[, whichLogged]), naturalspace)
  #     }
  #   }
  # } 
  # 
  # colnames(Paramset) <- names(Mle)  
  # toAdd = setdiff(c('range','shape','nugget','anisoRatio', 'anisoAngleRadians'), names(Mle))
  # otherParams = matrix(Model$opt$mle[toAdd], nrow=nrow(Paramset), ncol = length(toAdd),
  #                      dimnames = list(rownames(Paramset), toAdd), byrow=TRUE)
  # Params2 <- cbind(Paramset, otherParams)
  # 
  # result2<-gpuLik::getProfLogL(data= Model$data,
  #                                 formula=Model$model$formula,
  #                                 coordinates=Model$data@coords,
  #                                 params=Params2,
  #                                 boxcox = Model$parameters['boxcox'],
  #                                 type = "double",
  #                                 NparamPerIter=20,
  #                                 gpuElementsOnly=FALSE,
  #                                 reml=FALSE,
  #                                 Nglobal=c(128,64),
  #                                 Nlocal=c(16,16),
  #                                 NlocalCache=2000)
  # 
  # if(length(result1$boxcox)==2){
  #   A <- result2$LogLik[, 2]
  # }else{
  #   A <- result2$LogLik[, 1]
  # }
  # FirstDeri <- rep(0, length(Mle))
  # names(FirstDeri) <- names(MleGamma)
  # 
  # FirstDeri[1] <- (A[1] - A[2])/(2*deltas[1])
  # FirstDeri[2] <- (A[3] - A[4])/(2*deltas[2])
  # FirstDeri[3] <- (A[5] - A[6])/(2*deltas[3])
  # FirstDeri[4] <- (A[7] - A[8])/(2*deltas[4])
  # FirstDeri[5] <- (A[9] - A[10])/(2*deltas[5])
  # 
  # index <- which(abs(FirstDeri) > 0.01)
  # }
  # 
  # #Sigma <- solve(-HessianMat)
  # 
  # newMle <- Mle
  # newMleGamma <- MleGamma
  # 
  # 
  # if((('nugget' %in% names(Mle)) & Mle['nugget'] < delta) | ('shape' %in% names(Mle)) & Mle['shape'] >= kappa){
  # 
  # for(i in 1:length(index)){
  #   #newMle[index[i]] <- Mle[index[i]] + Sigma[index[i],index[i]]*FirstDeri[index[i]]
  #   newMleGamma[index[i]] <- MleGamma[index[i]] + solve(-HessianMat)[index[i],index[i]]*FirstDeri[index[i]]
  # } 
  # 
  # 
  #   if(('shape' %in% names(Mle)) & Mle['shape'] < 4){
  #     if(!('anisoRatio' %in% names(Mle))){
  #       newMle <- c(exp(newMleGamma[paste("log(", names(Mle)[whichLogged], ")",sep="")]), newMleGamma[-whichLogged])
  #     }else{
  #     temp <- as.data.frame(ParamsetGamma[,'aniso1'] + 1i * ParamsetGamma[,'aniso2'])
  #     if(Mle['anisoRatio'] <= 1){
  #       naturalspace <- cbind(1/(Mod(temp[,1])^2 + 1), Arg(temp[,1])/2 + pi/2)
  #       Paramset <- cbind(sqrt(exp(ParamsetGamma[,'sumLogRange'])*naturalspace[,1]), exp(ParamsetGamma[, whichLogged]), ParamsetGamma[,'nugget'], naturalspace)
  #     }else{
  #       naturalspace <- cbind(Mod(temp[,1])^2 + 1, Arg(temp[,1])/2)
  #       Paramset <- cbind(sqrt(exp(ParamsetGamma[,'sumLogRange'])*naturalspace[,1]), exp(ParamsetGamma[, whichLogged]), ParamsetGamma[,'nugget'], naturalspace)
  #     }
  #   }}else{
  #     if(!('anisoRatio' %in% names(Mle))){
  #       newMle <- c(exp(newMleGamma[paste("log(", names(Mle)[whichLogged], ")",sep="")]), newMleGamma[-whichLogged])
  #     }else{
  #     if(Mle['anisoRatio'] <= 1){
  #       naturalspace <- cbind(1/(Mod(temp[,1])^2 + 1), Arg(temp[,1])/2 + pi/2)
  #       Paramset <- cbind(sqrt(exp(ParamsetGamma[,'sumLogRange'])*naturalspace[,1]), (1/ParamsetGamma[, whichLogged])^2, ParamsetGamma[,'nugget'], naturalspace)
  #     }else{
  #       naturalspace <- cbind(Mod(temp[,1])^2 + 1, Arg(temp[,1])/2)
  #       Paramset <- cbind(sqrt(exp(ParamsetGamma[,'sumLogRange'])*naturalspace[,1]), (1/ParamsetGamma[, whichLogged])^2, ParamsetGamma[,'nugget'], naturalspace)
  #     }      
  #   }
  #   }
  # names(newMle) <- names(Mle)
  # 
  # ## check if newMle's are on boundary
  # # if('nugget' %in% names(Mle) & newMle['nugget'] < 0){
  # #   newMle['nugget'] = 0
  # # }
  # # 
  # # 
  # # if('shape' %in% names(Mle) & newMle['shape'] < 0){
  # #   newMle['shape'] = 0
  # # }
  # # 
  # # check mle shape
  # if('shape' %in% names(Mle) & newMle['shape'] >= kappa){
  #   newMle['shape'] = 1/newMle['shape']
  #   # if(Mle['shape'] <= delta)
  #   #   Mle['shape'] = delta
  #   #parToLog <- parToLog[!parToLog %in% 'shape']
  # }
  # }

  
  # if((('nugget' %in% names(Mle)) & Mle['nugget'] < delta) | ('shape' %in% names(Mle)) & Mle['shape'] >= kappa){
  # output = list(Gradiant= FirstDeri,
  #               HessianMat = HessianMat,
  #               #Sigma = Sigma,
  #               originalPoint = Mle,
  #               centralPoint = newMle,
  #               centralPointGamma = newMleGamma,
  #               parToLog = parToLog,
  #               whichAniso = whichAniso,
  #               data = cbind(Params1, result1$LogLik))
  # }else{
    output = list(HessianMat = HessianMat,
                  originalPoint = Mle,
                  parToLog = parToLog,
                  whichAniso = whichAniso,
                  data = cbind(Params1, result1$LogLik))  
  # }
  
  output
}

    
    
    
    
    
    

        
    
    
    
    
    
    
    
    


    
#' @title set loglikelihood locations
#' @useDynLib gpuLik
#' @export

    configParams <- function(Model,
                             alpha=c(0.001, 0.01, 0.1, 0.2, 0.5, 0.8, 0.95, 0.99),
                             logNugget = FALSE,
                             Mle = NULL,
                             boxcox = NULL,# a vector of confidence levels 1-alpha
                             shapeRestrict = 1000#,                             randomNugget = 0
    ){
      

      
      ## get the Hessian
      if(logNugget == TRUE){
      output <- getHessianLog(Model = Model,
                           Mle = Mle, 
                           boxcox = boxcox)
      }else{
       output <- getHessianNolog(Model = Model,
                           Mle = Mle, 
                           boxcox = boxcox) 
      }
   
      Mle <- output$originalPoint
      MleGamma <- output$centralPointGamma
      #Sigma <- output$Sigma
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

      
      if(logNugget == TRUE & Mle['shape'] < 4){

      if(length(Mle)==2){
        pointsSphere = exp(1i*seq(0, 2*pi, len=25))
        pointsSphere = pointsSphere[-length(pointsSphere)]
        pointsSphere2d = cbind(Re(pointsSphere), Im(pointsSphere))
        #plot(pointsSphere2d)
        if('anisoRatio' %in% names(Mle)){
          for(i in 1:length(alpha)){
            clevel <- stats::qchisq(1 - alpha[i], df = 2)
            pointsEllipseGammaspace = t(sqrt(clevel) * eig$vectors %*% diag(sqrt(eig$values)) %*%  t(pointsSphere2d) + MleGamma)
            colnames(pointsEllipseGammaspace) <- names(MleGamma)
            temp <- as.data.frame(pointsEllipseGammaspace[,'aniso1'] + 1i * pointsEllipseGammaspace[,'aniso2'])
            naturalspace <- cbind(Mod(temp[,1])^2 + 1, Arg(temp[,1])/2)
            pointsEllipse <- cbind(exp(pointsEllipseGammaspace[,whichLogged]),naturalspace)
            colnames(pointsEllipse) <- names(Mle)
            out_list[[i]] = pointsEllipse
          }         
        }else{
          for(i in 1:length(alpha)){
            clevel <- stats::qchisq(1 - alpha[i], df = 2)
            pointsEllipseGammaspace = t(sqrt(clevel) * eig$vectors %*% diag(sqrt(eig$values)) %*%  t(pointsSphere2d) + MleGamma)
            colnames(pointsEllipseGammaspace) <- names(MleGamma)
            pointsEllipse <- cbind(exp(pointsEllipseGammaspace[,paste("log(", names(Mle)[whichLogged], ")",sep="")]))
            colnames(pointsEllipse) <- names(Mle)
            out_list[[i]] = pointsEllipse
          } 
        }
      }else if(length(Mle)==3){
        #load('/home/ruoyong/gpuLik/data/coords3d.RData')
        #system.file("data","coords3d.RData", package = "gpuLik")
        coords3d <- gpuLik:::coords3d
        if('anisoRatio' %in% names(Mle)){
            for(i in 1:length(alpha)){
              clevel <- stats::qchisq(1 - alpha[i], df = 3)
              pointsEllipseGammaspace = t(sqrt(clevel) * eig$vectors %*% diag(sqrt(eig$values)) %*%  t(coords3d) + MleGamma)
              colnames(pointsEllipseGammaspace) <- names(MleGamma)
              temp <- as.data.frame(pointsEllipseGammaspace[,'aniso1'] + 1i * pointsEllipseGammaspace[,'aniso2'])
              naturalspace <- cbind(Mod(temp[,1])^2 + 1, Arg(temp[,1])/2)
              fixed = matrix(Model$opt$mle[fixedVar], nrow=nrow(pointsEllipseGammaspace), ncol = length(fixedVar),
                             dimnames = list(rownames(pointsEllipseGammaspace), fixedVar), byrow=TRUE)
              pointsEllipse <- cbind(sqrt(exp(pointsEllipseGammaspace[,'sumLogRange'])*naturalspace[,1]), exp(pointsEllipseGammaspace[,whichLogged]),naturalspace,fixed)
              colnames(pointsEllipse) <- names(Mle)
              out_list[[i]] = pointsEllipse
            }
          # }
        }else{
          for(i in 1:length(alpha)){
            clevel <- stats::qchisq(1 - alpha[i], df = 3)
            pointsEllipseGammaspace = t(sqrt(clevel) * eig$vectors %*% diag(sqrt(eig$values)) %*%  t(coords3d) + MleGamma)
            colnames(pointsEllipseGammaspace) <- names(MleGamma)
            pointsEllipse <- cbind(exp(pointsEllipseGammaspace[,paste("log(", names(Mle)[whichLogged], ")",sep="")]))
            colnames(pointsEllipse) <- names(Mle)
            out_list[[i]] = pointsEllipse
          }
          }
        }else if(length(Mle)==4){
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
              pointsEllipse <- cbind(sqrt(exp(pointsEllipseGammaspace[,'sumLogRange'])*naturalspace[,1]), exp(pointsEllipseGammaspace[,whichLogged]),naturalspace,fixed)
              colnames(pointsEllipse) <- names(Mle)
              out_list[[i]] = pointsEllipse
            }
        # }
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
            pointsEllipse <- cbind(sqrt(exp(pointsEllipseGammaspace[,'sumLogRange'])*naturalspace[,1]), exp(pointsEllipseGammaspace[,whichLogged]),naturalspace)
            colnames(pointsEllipse) <- names(Mle)
            out_list[[i]] = pointsEllipse
          }
        # }
      }
      }
      
      
      if(logNugget == TRUE & Mle['shape'] >= 4){

        if(length(Mle)==2){
          pointsSphere = exp(1i*seq(0, 2*pi, len=25))
          pointsSphere = pointsSphere[-length(pointsSphere)]
          pointsSphere2d = cbind(Re(pointsSphere), Im(pointsSphere))
          #plot(pointsSphere2d)
          if('anisoRatio' %in% names(Mle)){
            for(i in 1:length(alpha)){
              clevel <- stats::qchisq(1 - alpha[i], df = 2)
              pointsEllipseGammaspace = t(sqrt(clevel) * eig$vectors %*% diag(sqrt(eig$values)) %*%  t(pointsSphere2d) + MleGamma)
              colnames(pointsEllipseGammaspace) <- names(MleGamma)
              temp <- as.data.frame(pointsEllipseGammaspace[,'aniso1'] + 1i * pointsEllipseGammaspace[,'aniso2'])
              naturalspace <- cbind(Mod(temp[,1])^2 + 1, Arg(temp[,1])/2)
              pointsEllipse <- cbind((1/pointsEllipseGammaspace[, whichLogged])^2,naturalspace)
              colnames(pointsEllipse) <- names(Mle)
              out_list[[i]] = pointsEllipse
            }
          }else{
            for(i in 1:length(alpha)){
              clevel <- stats::qchisq(1 - alpha[i], df = 2)
              pointsEllipseGammaspace = t(sqrt(clevel) * eig$vectors %*% diag(sqrt(eig$values)) %*%  t(pointsSphere2d) + MleGamma)
              colnames(pointsEllipseGammaspace) <- names(MleGamma)
              pointsEllipse <- cbind((1/pointsEllipseGammaspace[, whichLogged])^2)
              colnames(pointsEllipse) <- names(Mle)
              out_list[[i]] = pointsEllipse
            }
          }
        }else if(length(Mle)==3){
          #load('/home/ruoyong/gpuLik/data/coords3d.RData')
          #system.file("data","coords3d.RData", package = "gpuLik")
          coords3d <- gpuLik:::coords3d
          if('anisoRatio' %in% names(Mle)){
              for(i in 1:length(alpha)){
                clevel <- stats::qchisq(1 - alpha[i], df = 3)
                pointsEllipseGammaspace = t(sqrt(clevel) * eig$vectors %*% diag(sqrt(eig$values)) %*%  t(coords3d) + MleGamma)
                colnames(pointsEllipseGammaspace) <- names(MleGamma)
                temp <- as.data.frame(pointsEllipseGammaspace[,'aniso1'] + 1i * pointsEllipseGammaspace[,'aniso2'])
                naturalspace <- cbind(Mod(temp[,1])^2 + 1, Arg(temp[,1])/2)
                fixed = matrix(Model$opt$mle[fixedVar], nrow=nrow(pointsEllipseGammaspace), ncol = length(fixedVar),
                               dimnames = list(rownames(pointsEllipseGammaspace), fixedVar), byrow=TRUE)
                pointsEllipse <- cbind(sqrt(exp(pointsEllipseGammaspace[,'sumLogRange'])*naturalspace[,1]), (1/pointsEllipseGammaspace[, whichLogged])^2,naturalspace,fixed)
                colnames(pointsEllipse) <- names(Mle)
                out_list[[i]] = pointsEllipse
              }
          }else{
            for(i in 1:length(alpha)){
              clevel <- stats::qchisq(1 - alpha[i], df = 3)
              pointsEllipseGammaspace = t(sqrt(clevel) * eig$vectors %*% diag(sqrt(eig$values)) %*%  t(coords3d) + MleGamma)
              colnames(pointsEllipseGammaspace) <- names(MleGamma)
              fixed = matrix(Model$opt$mle[fixedVar], nrow=nrow(pointsEllipseGammaspace), ncol = length(fixedVar),
                             dimnames = list(rownames(pointsEllipseGammaspace), fixedVar), byrow=TRUE)
              pointsEllipse <- cbind((1/pointsEllipseGammaspace[, whichLogged])^2, fixed)
              colnames(pointsEllipse) <- names(Mle)
              out_list[[i]] = pointsEllipse
            }
          }
        }else if(length(Mle)==4){
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
              pointsEllipse <- cbind(sqrt(exp(pointsEllipseGammaspace[,'sumLogRange'])*naturalspace[,1]), (1/pointsEllipseGammaspace[, whichLogged])^2,naturalspace,fixed)
              colnames(pointsEllipse)[1:4] <- names(Mle)
              out_list[[i]] = pointsEllipse
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
              pointsEllipse <- cbind(sqrt(exp(pointsEllipseGammaspace[,'sumLogRange'])*naturalspace[,1]), (1/pointsEllipseGammaspace[, whichLogged])^2,naturalspace)
              colnames(pointsEllipse) <- names(Mle)
              out_list[[i]] = pointsEllipse
            }
        }
      }
      
      
      if(logNugget == FALSE & ('shape' %in% names(Mle)) & Mle['shape'] < 4){
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
              out_list[[i]] = pointsEllipse
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
              out_list[[i]] = pointsEllipse
            }
        }
      }
      
      
      if(logNugget == FALSE & ('shape' %in% names(Mle)) & Mle['shape'] >= 4){
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
              out_list[[i]] = pointsEllipse
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
              out_list[[i]] = pointsEllipse
            }
        }
      }
      
      if(logNugget == FALSE & !('shape' %in% names(Mle)) & length(Mle)==4){
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
              out_list[[i]] = pointsEllipse
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
             vector[j] = stats::runif(1, 0, 3) #-vector[j]#
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
      
      
      names(out_list) <- paste0("alpha", alpha, sep="")
      
      
      out_list[[length(alpha)+1]] <- boxcoxSetup
    
      out_list
      
      
   }
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
       
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    