#' @title maternGpuParam
#' @description Create parameters be used by maternBatch on GPU
#' @param x R matrix of covariance parameters 
#' @param type precision type of paramters, 'double' or 'float'
#' @useDynLib gpuLik
#' @export



maternGpuParam = function(x, type='double') {

  
  if(is.vector(x)) x=matrix(x, nrow=1, dimnames = list(1, names(x)))
  
  anisoParams = c('anisoAngleDegrees','anisoAngleRadians')
  if(sum(anisoParams %in% colnames(x)) == 1) {
    #only one of radians and degrees
    if('anisoAngleDegrees' %in% colnames(x)) {
      x = cbind(x, anisoAngleRadians = 2*pi*x[,'anisoAngleDegrees']/360)
    } else {
      x = cbind(x, anisoAngleDegrees = 360*x[,'anisoAngleRadians']/(2*pi))
    }
  }
  theColumns = c('range','shape','variance','nugget','anisoRatio','anisoAngleRadians','anisoAngleDegrees')
  if(all(theColumns %in% colnames(x))) {
    x = x[,theColumns]
  } else {
    if(requireNamespace("geostatsp")) {
      x = geostatsp::fillParam(x)
    } else {
      stop("can't process model parameters, install the geostatsp package")   
    }
  }
    
  paramsGpu = vclMatrix(cbind(x, matrix(0, nrow(x), 22-ncol(x))), 
                        type=type)
  
  gpuR::colnames(paramsGpu) = colnames(x)
  
  paramsGpu
}





























