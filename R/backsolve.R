#' @title backsolveBatch
#' @description solve A * C = B for C on a GPU, where A, B, C are batches of square matrices
#' @param C an object of class 'vclMatrix'
#' @param A an object of class 'vclMatrix', upper triangular values are 0
#' @param B an object of class 'vclMatrix', batches of rectangular matrices
#' @param numbatchB number of batches in B, if 1 then B has the same matrix for all matrices in A batch
#' @param diagIsOne logical, indicates if all the diagonal entries in A matrices are 1
#' @param Nglobal Size of the index space for use
#' @param Nlocal Work group size of the index space
#' @param NlocalCache local memory cached
#' @param Cstartend a vector that selects the range of C, c(startrow, numberofrows, startcolumn, numberofcols), row starts from 0
#' @param Astartend a vector that selects the range of A 
#' @param Bstartend a vector that selects the range of B
#' @param verbose if TRUE, print out more information
#' @note result matrices are stored in C respectively, no returned objects
#' @useDynLib gpuBatchMatrix
#' @export



backsolveBatch <- function(C, 
                           A,  # must be batches of square matrices
                           B,  #vclmatrices
                           numbatchB, #sometimes B can have only 1 batch, for repeated same batches
                           diagIsOne,
                           Nglobal, 
                           Nlocal, 
                           NlocalCache,
                           Cstartend,
                           Astartend,
                           Bstartend,
                           verbose=FALSE){

  
  nbatch<-nrow(A)/ncol(A)

  if(missing(Cstartend)) {
    Cstartend=c(0, nrow(C)/nbatch, 0, ncol(C))
  }
  
  if(missing(Astartend)) {
    Astartend=c(0, nrow(A)/nbatch, 0, ncol(A))
  }
  
  if(missing(Bstartend)) {
    Bstartend=c(0, nrow(B)/numbatchB, 0, ncol(B))
   }
  

  
  
  
  
  
  if(verbose){ message(paste('global work items', Nglobal,
                             'local work items', Nlocal))}


 

  backsolveBatchBackend(C, A, B, 
                        Cstartend, Astartend, Bstartend, 
                        numbatchB, diagIsOne, 
                        Nglobal, Nlocal, NlocalCache)



  }

  