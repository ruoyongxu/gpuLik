#' @title cholBatch
#' @description performs Cholesky decomposition in batches on A = L D' L^t on a GPU
#' @param A a vclMatrix on GPU, positive definite
#' @param D a vclMatrix on GPU, each row contains diagonal elements of D' 
#' @param numbatchD number of batches
#' @param Nglobal Size of the index space for use
#' @param Nlocal Work group size of the index space
#' @param NlocalCache local memory cached
#' @param Astartend a vector that selects the range of A, c(startrow, numberofrows, startcolumn, numberofcols), row starts from 0
#' @param Dstartend a vector that selects the range of D
#' @param verbose if TRUE, print out more information
#' @note computed L and D' are stored in A and D respectively, no returned objects
#' @useDynLib gpuBatchMatrix
#' @export



cholBatch <- function(A,
                      D,
                      numbatchD,
                      Nglobal,
                      Nlocal,   # needs Nglobal[2]=Nlocal[2]
                      NlocalCache = gpuR::gpuInfo()$localMem/32,
                      Astartend,
                      Dstartend,
                      verbose=FALSE){
  
  
  if(missing(Astartend) & missing(numbatchD) & missing(D)){
    Astartend = c(0, ncol(A), 0, ncol(A))
    numbatchD = nrow(A)/ncol(A)
    D = vclMatrix(0, numbatchD, ncol(A), type = gpuR::typeof(A))
  }
  
  if(missing(numbatchD) )
     numbatchD=nrow(D)
     
  if(missing(D)) 
    D = vclMatrix(0, numbatchD, ncol(A), type = gpuR::typeof(A))
  
  if(missing(Astartend)) 
    Astartend=c(0, nrow(A)/numbatchD, 0, ncol(A))
  
  if(missing(Dstartend)) {
    Dstartend=c(0, numbatchD, 0, ncol(D))
  }
  
  
   if(Nlocal[2]!=Nglobal[2]){
     warning("local and global work sizes should be identical for dimension 2, ignoring global")
     Nglobal[2]=Nlocal[2]
   }
  
 
  
  if(verbose){ message(paste('global work items', Nglobal,
                             'local work items', Nlocal))}
  
  
  
  
  cholBatchBackend(A, D, 
                   Astartend, Dstartend, 
                   numbatchD,
                   Nglobal, Nlocal, NlocalCache)
  
   #theResult = list(L=A, diag=D)


   #theResult
   invisible(D)
  
  
}