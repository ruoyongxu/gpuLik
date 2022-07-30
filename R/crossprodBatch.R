#' @title crossprodBatch
#' @description computes C = t(A) A or t(A) D' A or t(A) D'^(-1) A in batches on a GPU
#' @param C a vclMatrix on GPU, square matrices batches
#' @param A a vclMatrix on GPU, rectangular matrix batches
#' @param D a vclMatrix on GPU, rectangular matrix batches, rows are diagonals of D', stacked row-wise 
#' @param invertD set to 1 for C = t(A) D^(-1) A
#' @param Nglobal vector of number of global work items
#' @param Nlocal vector of number of local work items
#' @param NlocalCache elements in local cache
#' @param Cstartend a vector (startrow, nRow, startcol,nCol) that indicates the selected part of each submatrix in C
#' @param Astartend a vector (startrow, nRow, startcol,nCol) that indicates the selected part of each submatrix in A
#' @param Dstartend a vector (startrow, nRow, startcol,nCol) that indicates the selected part of each submatrix in D
#' @note computed results are stored in C, no returned objects
#' @useDynLib gpuLik
#' @export


 
crossprodBatch <- function(C,   # must be batch of square matrices 
                           A,
                           D,
                           invertD,
                           Nglobal, 
                           Nlocal, 
                           NlocalCache,
                           Cstartend, Astartend, Dstartend) {
  
  Nbatches = nrow(C)/ncol(C)
  
  if(missing(Cstartend)) {
    Cstartend=c(0, ncol(C), 0, ncol(C))
  }
  
  if(missing(Astartend)) {
    Astartend=c(0, nrow(A)/Nbatches, 0, ncol(A))
  }
  
  if(missing(Dstartend)) {
    Dstartend=c(0, 1, 0, ncol(D))
  }
  
  
  if((NlocalCache - Nlocal[1]*Nlocal[2])<0){
    warning("a larger NlocalCache required")
  }
  
  crossprodBatchBackend(C,A,D,invertD,Cstartend,Astartend,Dstartend, Nglobal,Nlocal, NlocalCache)
  
  invisible()
  
  
}




