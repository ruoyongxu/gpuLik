#' @title multiplyLowerDiagonalBatch
#' @description computes output = LD'B in batches on a GPU
#' @param L  lower triangular matrices in batches
#' @param D  diagonal matrices in batches, each row contains diagonal elements of D' 
#' @param B  matrices in batches
#' @param diagIsOne logical, whether the diagonal of L is one 
#' @param transformD how to transform D, can be any OpenCL C built-in math function
#' @param output the result of LD'B
#' @param Nglobal the size of the index space for use
#' @param Nlocal the work group size of the index space 
#' @param NlocalCache a number
#' 
#' @useDynLib gpuBatchMatrix
#' @export






multiplyLowerDiagonalBatch <- function(L, D, B, output,# output = L  D B, L lower triangular, D diagonal
                      diagIsOne, # diagonal of L is one
                      transformD, 
                      Nglobal,
                      Nlocal,
                      NlocalCache){
  
  if(missing(output)) {
    output = vclMatrix(0, nrow(L), ncol(B), 
                    type = gpuR::typeof(L))
  }  
  
  multiplyLowerDiagonalBatchBackend(
               output,
               L,
               D,
               B,
               diagIsOne,
               transformD,
               Nglobal,
               Nlocal,
               NlocalCache)
  
  invisible(output)
  
}











