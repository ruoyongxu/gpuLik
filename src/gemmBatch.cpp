/*
 Rcpp::IntegerVector workgroupSize,   // global 0 1 2, local 0 1 2
 Rcpp::IntegerVector transposeABC, // transposeA, transposeB, transposeC
 Rcpp::IntegerVector submatrixA, // [rowStart, nRowsSub, nRowsTotal, colStart, ...]
 Rcpp::IntegerVector submatrixB,
 Rcpp::IntegerVector submatrixC,
 Rcpp::IntegerVector batches, // nRow, nCol, recycleArow, recycleAcol, recycleB row col
 Rcpp::IntegerVector NlocalCache,  //cacheSizeA, cacheSizeB, 
 
 NOTE: recycling A, B not implemented 
 */

/*
 * submatrixA = 
 *  [rowStartA, nRowsAsub, nRowsA, colStartA, nColsAsub, nColsA]
 *  indexing from zero
 *  submatrix[DmatrixRow, DmatrixCol] of A is
 *  A[seq(nRowsA * DmatrixRow + rowStartA, len=nRowsAsub),
 *  seq(nColsA * DmatrixCol + colStartA, len=nColsAsub)]
 *  
 * batches = 
 * nRowBatch, nColBatch, recycleArow, recycleAcol, recycleBrow, recycleBcol
 * recycleArow = 1, there are no row batches for A, use the same A for all batches
 * 
 * Rcpp::IntegerVector workgroupSize, global 0 1 2, local 0 1 2
 * global0 is row batch, col batch not parallelized
 * global1 and global2 are rows and columns of C
 * local0 is ignored, set to 1.  group 1 and group2 cache rows and columns of A and B
 * DmatrixRow = global0; DmatrixCol = not parallel, 
 * Drow = global1, Dcol = global2
 * 
 * 
 * A[DmatrixRow, DmatrixCol][Drow, Dcol] =
 * A[DmatrixRow * (NpadA * nRowsTotal) + 
 *     NpadA * (rowStart + Drow) + 
 *     DmatrixCol * nColsTotal +
 *     Dcol]
 * 
 */

#include "gpuRandom.hpp"
#include <algorithm>        // std::min
using namespace Rcpp;


//#define DEBUG


template <typename T> 
std::string gemmBatch2String(
    const int onlyDiagC,         // compute only the diagonal values of C
    Rcpp::IntegerVector transposeABC,  
    Rcpp::IntegerVector submatrixA,  
    Rcpp::IntegerVector submatrixB,
    Rcpp::IntegerVector submatrixC,
    Rcpp::IntegerVector recycle,      //"nCol", "recycleArow", "recycleAcol", "recycleBrow", "recycleBcol"
    Rcpp::IntegerVector NlocalCache,
    int NpadA, int NpadB, int NpadC) { 
  
  std::string typeString = openclTypeString<T>();
  std::string result = "";
  
  
  if(typeString == "double") {
    result += "\n#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\n";
  }
  
  
  result +=  
    //  "#define NmatrixRow " + std::to_string(batches[0]) + "\n"
    "#define NmatrixCol " + std::to_string(recycle[0]) + "\n"; 
  
  
  result +=  
    "#define NpadA "+ std::to_string(NpadA) + "\n"  
    "#define rowStartA " + std::to_string(submatrixA[0]) + "\n"
    "#define NrowTotalA " + std::to_string(submatrixA[2]) + "\n"
    "#define NpadNrowTotalA "+ std::to_string(NpadA * submatrixA[2]) + "\n"  
    "#define colStartA " + std::to_string(submatrixA[3]) + "\n"
    "#define NcolTotalA " + std::to_string(submatrixA[5]) + "\n\n";
  
  result +=
    "#define NpadB "+ std::to_string(NpadB) + "\n"  
    "#define rowStartB " + std::to_string(submatrixB[0]) + "\n"
    "#define NrowTotalB " + std::to_string(submatrixB[2]) + "\n"
    "#define NpadNrowTotalB "+ std::to_string(NpadB * submatrixB[2]) + "\n"  
    "#define colStartB " + std::to_string(submatrixB[3]) + "\n"
    "#define NcolTotalB " + std::to_string(submatrixB[5]) + "\n\n";
  
  result += 
    "#define NpadC "+ std::to_string(NpadC) + "\n"
    "#define rowStartC " + std::to_string(submatrixC[0]) + "\n";
  
  if(onlyDiagC){
    result +=
      "#define NrowTotalC 1 \n"
      "#define NpadNrowTotalC "+ std::to_string(NpadC * 1) + "\n"
      "#define MIN(x, y) (x < y ? x : y) \n";
  }else{
    result +=
      "#define NrowTotalC " + std::to_string(submatrixC[2]) + "\n"
      "#define NpadNrowTotalC "+ std::to_string(NpadC * submatrixC[2]) + "\n";  
  }
  
  result +=
    "#define colStartC " + std::to_string(submatrixC[3]) + "\n"
    "#define NcolTotalC " + std::to_string(submatrixC[5]) + "\n\n";
  
  
  if(transposeABC[0]) { // transpose A, Ninner is rows of A^T, rows of A， A is Nrow by Ninner
    result += 
      "#define Nrow " + std::to_string(submatrixA[4]) + "\n"  
      "#define Ninner " + std::to_string(submatrixA[1]) + "\n"
      "#define Ai0orig(ii) (DmatrixRow * NpadNrowTotalA + NpadA * (rowStartA + (ii)) + DmatrixCol * NcolTotalA + colStartA + DrowBlock)\n"
      "#define Ai0(ii) (startInnerA + NpadA * (ii) )\n";
  } else { // no transpose, Ninner is cols of A
    result += 
      "#define Nrow " + std::to_string(submatrixA[1]) + "\n"  
      "#define Ninner " + std::to_string(submatrixA[4]) + "\n"
      "#define Ai0orig(ii) (DmatrixRow * NpadNrowTotalA + NpadA * (rowStartA + DrowBlock) + DmatrixCol * NcolTotalA + colStartA + (ii))\n"
      "#define Ai0(ii) (startInnerA + (ii) )\n";
  }
  result += "\n";
  
  if(transposeABC[1]) { // transpose B
    result += 
      "#define Ncol " + std::to_string(submatrixB[1]) + "\n"  
      "#define Bi0orig(ii) ( DmatrixRow * NpadNrowTotalB + NpadB * (rowStartB + DcolBlock) + DmatrixCol * NcolTotalB + colStartB + (ii))\n"
      "#define Bi0(ii) (startInnerB + (ii) )\n\n";
  } else { // no transpose, columns of B
    result += 
      "#define Ncol " + std::to_string(submatrixB[4]) + "\n"  
      "#define Bi0orig(ii) ( DmatrixRow * NpadNrowTotalB + NpadB * (rowStartB + (ii)) + DmatrixCol * NcolTotalB + colStartB + DcolBlock)\n"
      "#define Bi0(ii) (startInnerB + NpadB * (ii) )\n\n";
  }
  result += "\n";
  
  if(transposeABC[2] && !onlyDiagC) { // transpose C, Nrow is rows of C^T, cols of C
    result += 
      "#define Cijorig ( DmatrixRow * NpadNrowTotalC + NpadC * (colStartC + Dcol) + DmatrixCol * NrowTotalC + rowStartC + Drow)\n"
      "#define Cij (startMatrixC+ NpadC * Dcol + Drow)\n\n";
  } else if(!transposeABC[2] && !onlyDiagC){  // no transpose
    result += 
      "#define Cijorig ( DmatrixRow * NpadNrowTotalC + NpadC * (rowStartC + Drow) + DmatrixCol * NcolTotalC + colStartC + Dcol)\n"
      "#define Cij (startMatrixC+ NpadC * Drow + Dcol)\n\n";
  }else if(transposeABC[2] && onlyDiagC){
    result += 
      "#define Cijorig ( DmatrixRow * NpadNrowTotalC + NpadC * (colStartC + Dcol) + DmatrixCol * NrowTotalC + rowStartC + Drow)\n"
      "#define Cij (startMatrixC  + Drow)\n\n";
  }else if(!transposeABC[2] && onlyDiagC){
    result += 
      "#define Cijorig ( DmatrixRow * NpadNrowTotalC + NpadC * (rowStartC + Drow) + DmatrixCol * NcolTotalC + colStartC + Dcol)\n"
      "#define Cij (startMatrixC + Drow)\n\n";
  }
  
  
  result += "\n";
  
  result +=  
    "#define cacheSizeA " + std::to_string(NlocalCache[0]) + "\n"
    "#define cacheSizeB " + std::to_string(NlocalCache[1]) + "\n"; 
  
  
  
  
  
  result += "\n__kernel void gemm( __global "  + typeString+ "* A,\n"
  " __global "  + typeString+ "* B,\n"
  " __global " + typeString+ "* C,\n"
  "   int NrowStartC,\n"
  "   int NmatrixRow){   //nRowBatch \n\n";
  
  if(onlyDiagC){
    result +=
      "const int NrowStop = MIN(Nrow, Ncol);\n\n";
  }
  
  result += typeString + " acc;\n"
  "int DmatrixRow, DmatrixCol;\n"
  "int startMatrixA, startMatrixB, startMatrixC, startInnerA, startInnerB;\n"
  "int Drow, DrowBlock, Dcol, DcolBlock;\n"
  "int Dinnerp1;\n";
  result += "event_t wait;\n\n";
  
  result += "local " + typeString + 
    " localCacheA1[cacheSizeA], localCacheB1[cacheSizeB];\n";   
  result += "local " + typeString + 
    " localCacheA2[cacheSizeA], localCacheB2[cacheSizeB];\n";
  result += "local " + typeString + 
    " *Anow, *Anext, *Atemp, *Bnow, *Bnext, *Btemp;\n";   
  
  result += "\n";
  result += 
    " for(DmatrixRow = get_global_id(0);\n" 
    "   DmatrixRow < NmatrixRow;\n" 
    "   DmatrixRow += get_global_size(0)) {\n"
    
    " for(DmatrixCol = 0;\n"
    "   DmatrixCol < NmatrixCol;\n" 
    "   DmatrixCol ++) {\n\n";
  
  result += "startMatrixA = DmatrixRow * NpadNrowTotalA + NpadA * rowStartA + DmatrixCol * NcolTotalA + colStartA;\n";
  if(recycle[3]) {
    result += "startMatrixB = NpadB * rowStartB + DmatrixCol * NcolTotalB + colStartC;\n";
  } else {
    result += "startMatrixB = DmatrixRow * NpadNrowTotalB + NpadB * rowStartB + DmatrixCol * NcolTotalB + colStartB;\n";
  }
  
  result += "startMatrixC = (DmatrixRow + NrowStartC) * NpadNrowTotalC + NpadC * rowStartC + DmatrixCol * NcolTotalC + colStartC;\n";
  
  
  
  
  
  result +=
    "   for(DrowBlock = get_group_id(1)*get_local_size(1);\n"; // row for work item ?,0,?
  
  if(onlyDiagC){
    result +=   "   DrowBlock < NrowStop;\n";
  }else{
    result +=
    "   DrowBlock < Nrow;\n";
  }
  
  result +=
    "     DrowBlock += get_global_size(1)) {\n"
    "     Drow = DrowBlock + get_local_id(1);\n";
  
  
  if(onlyDiagC){
    result +=
      "       Dcol = Drow;\n"
      "       DcolBlock = Dcol;\n";
  }else{
    result +=
    "   for(DcolBlock = get_group_id(2)*get_local_size(2);\n" // col for work item ?,?,0
    "     DcolBlock < Ncol;\n"
    "     DcolBlock += get_global_size(2)) {\n"
    "     Dcol = DcolBlock + get_local_id(2);\n";
  }
  
  
  
  
  if(transposeABC[0]) {  
    result += "startInnerA = startMatrixA + DrowBlock ;\n";
  } else {
    result += "startInnerA = startMatrixA + NpadA * DrowBlock;\n";
  }
  if(transposeABC[1]) {  
    result += "startInnerB = startMatrixB + NpadB * DcolBlock ;\n";
  } else {
    result += "startInnerB = startMatrixB + DcolBlock;\n";
  }
  
  
  
  result += "acc = 0.0;\n";
  result += "wait =  (event_t) 0;\n";
  
  result += "Anow = localCacheA1;\n"
  "Anext = localCacheA2;\n"
  "Bnow = localCacheB1;\n"
  "Bnext = localCacheB2;\n";
  
  // cache A[1:Nlocal, 1], and B[1, 1:Nlocal]
  if(transposeABC[0] ) {   // transpose A
    result +=  "  wait = async_work_group_copy(\n"
    "    Anext, &A[Ai0(0)],\n"
    "    get_local_size(1), (event_t) 0);\n";
    
  } else {    // don't transpose A
    result +=  "  wait = async_work_group_strided_copy(\n"
    "    Anext, &A[Ai0(0)],\n"
    "    get_local_size(1), NpadA, (event_t) 0);\n";
    
  }
  
  
  
  
  if(transposeABC[1]) {
    result +=  "  wait = async_work_group_strided_copy(\n"
    "    Bnext, &B[Bi0(0)],\n"
    "    get_local_size(2), NpadB, wait);\n";
  } else {
    result +=  "  wait = async_work_group_copy(\n"
    "    Bnext, &B[Bi0(0)],\n"
    "    get_local_size(2), wait);\n";
  }
  
  
  
  
  
  
  
  
  
  result += "  wait_group_events (1, &wait);\n";
  
  result += // loop through remaining inner
    "   for(Dinnerp1=1;\n" 
    "     Dinnerp1 < Ninner;\n"
    "     Dinnerp1++) {\n";
  // cache row Dinner of C, col Dinner of A
  
  result +=  " Atemp = Anow;\n"
  " Btemp = Bnow;\n"
  " Anow = Anext;\n"
  " Bnow = Bnext;\n"
  " Anext = Atemp;\n"
  " Bnext = Btemp;\n";
  
  
  
  
  // cache ahead
  if(transposeABC[0]) {
    result +=  "  wait = async_work_group_copy(\n"
    "    Anext, &A[Ai0(Dinnerp1)],\n"
    "    get_local_size(1), (event_t) 0);\n";
  } else if(!transposeABC[0]){
    result +=  "  wait = async_work_group_strided_copy(\n"
    "    Anext, &A[Ai0(Dinnerp1)],\n"
    "    get_local_size(1), NpadA, (event_t) 0);\n";
  }
  
  
  
  if(transposeABC[1]) {
    result +=  "  wait = async_work_group_strided_copy(\n"
    "    Bnext, &B[Bi0(Dinnerp1)],\n"
    "    get_local_size(2), NpadB, wait);\n";
  } else {
    result +=  "  wait = async_work_group_copy(\n"
    "    Bnext, &B[Bi0(Dinnerp1)],\n"
    "    get_local_size(2), wait);\n";
  }
  //  "  barrier(CLK_LOCAL_MEM_FENCE);\n";
  
  
  
  
  result += "\n";
  // compute component from previously cached values
  // this will happen at the same time the caching is done
  
  
  
  result += "acc += Anow[get_local_id(1)] * "
  " Bnow[get_local_id(2)];\n";
  
  result += "\n";
  result += "  wait_group_events (1, &wait);\n";
  
  result += 
    "   }//Dinner Ninner\n";
  result += "\n";
  // last row of B, Dinner = Ninner
  // A and B are cached, don't need to cache ahead
  
  
  result += "acc += Anext[get_local_id(1)] * "
  " Bnext[get_local_id(2)];\n";
  
  
  result += "\n";
  result += 
    "\nif(Drow < Nrow & Dcol < Ncol) {\n"
    "  C[Cij] = acc;\n"
    "}\n\n";
  
  
  
  if(onlyDiagC){
  }else{
    result += 
      "   }//DcolBlock\n"; 
  }
  
  
  result +=
    "   }//DrowBlock\n";
  
  
  result += 
    " }//DmatrixRow\n"
    " }//DmatrixCol\n";
  result += 
    " }//kernel\n";
  
  
  return(result);
}








template <typename T> 
int gemmBatch2(
    viennacl::matrix<T> &A,
    viennacl::matrix<T> &B,
    viennacl::matrix_base<T> &C,
    Rcpp::IntegerVector transposeABC,  
    Rcpp::IntegerVector submatrixA,
    Rcpp::IntegerVector submatrixB,
    Rcpp::IntegerVector submatrixC,  
    Rcpp::IntegerVector batches, 
    Rcpp::IntegerVector workgroupSize,
    Rcpp::IntegerVector NlocalCache, 
    const int verbose,
    const int ctx_id) {
  
  
  
  
  
  
  Rcpp::IntegerVector integer = {1,2,3,4,5};
  Rcpp::IntegerVector recycle = batches[integer];      //"nCol", "recycleArow", "recycleAcol", "recycleBrow", "recycleBcol"
  
  std::string gemmString = gemmBatch2String<T>(
    0,   // onlyDiagC,         // compute only the diagonal values of C
    transposeABC,  
    submatrixA,
    submatrixB,
    submatrixC,
    recycle, 
    NlocalCache, 
    (int) A.internal_size2(),
    (int) B.internal_size2(),
    (int) C.internal_size2()
  );
  
  if(verbose)  Rcpp::Rcout << "\n\n" << gemmString << "\n\n";
  
  viennacl::ocl::context ctx(viennacl::ocl::get_context(ctx_id));
  viennacl::ocl::program & my_prog = ctx.add_program(gemmString, "mykernel");
  viennacl::ocl::kernel & gemmKernel = my_prog.get_kernel("gemm");
  
  gemmKernel.global_work_size(0, workgroupSize[0]);
  gemmKernel.global_work_size(1, workgroupSize[1]);
  gemmKernel.global_work_size(2, workgroupSize[2]);
  //  gemmKernel.local_work_size(0, workgroupSize[3]);
  gemmKernel.local_work_size(0, 1L);  // local size 0 must be 1
  gemmKernel.local_work_size(1, workgroupSize[4]);
  gemmKernel.local_work_size(2, workgroupSize[5]);  
  
  viennacl::ocl::enqueue(gemmKernel(A, B, C, 0, batches[0]));
  
  return 0;
}


template <typename T> 
SEXP gemmBatch2Typed(Rcpp::S4 AR,
                     Rcpp::S4 BR,
                     Rcpp::S4 CR,
                     Rcpp::IntegerVector transposeABC,  
                     Rcpp::IntegerVector submatrixA,
                     Rcpp::IntegerVector submatrixB,
                     Rcpp::IntegerVector submatrixC,  
                     Rcpp::IntegerVector batches, 
                     Rcpp::IntegerVector workgroupSize,
                     Rcpp::IntegerVector NlocalCache, 
                     const int verbose) {
  
  
  const int ctx_id = INTEGER(CR.slot(".context_index"))[0]-1;
  const bool BisVCL=1;
  int result;
  
  std::shared_ptr<viennacl::matrix<T> > A = getVCLptr<T>(AR.slot("address"), BisVCL, ctx_id);
  std::shared_ptr<viennacl::matrix<T> > B = getVCLptr<T>(BR.slot("address"), BisVCL, ctx_id);
  std::shared_ptr<viennacl::matrix<T> > C = getVCLptr<T>(CR.slot("address"), BisVCL, ctx_id);
  
  result = gemmBatch2<T>(*A, *B, *C, transposeABC, 
                         submatrixA, submatrixB, submatrixC, batches, 
                         workgroupSize, NlocalCache, verbose, ctx_id);  
  
  return Rcpp::wrap(result);
}





// [[Rcpp::export]]
SEXP gemmBatch2backend(
    Rcpp::S4 A,
    Rcpp::S4 B,  
    Rcpp::S4 C,
    Rcpp::IntegerVector transposeABC,  
    Rcpp::IntegerVector submatrixA,
    Rcpp::IntegerVector submatrixB,
    Rcpp::IntegerVector submatrixC, 
    Rcpp::IntegerVector batches, 
    Rcpp::IntegerVector workgroupSize,   
    Rcpp::IntegerVector NlocalCache,
    const int verbose) {
  
  SEXP result;
  
  //#ifdef UNDEF  
  Rcpp::traits::input_parameter< std::string >::type classVarR(RCPP_GET_CLASS(C));
  std::string precision_type = (std::string) classVarR;
  if(precision_type == "fvclMatrix") {
    result=gemmBatch2Typed<float>(A, B, C, 
                                  transposeABC, 
                                  submatrixA, submatrixB, submatrixC, batches, 
                                  workgroupSize, NlocalCache, verbose);
  } else if (precision_type == "dvclMatrix") {
    result=gemmBatch2Typed<double>(A, B, C, 
                                   transposeABC, 
                                   submatrixA, submatrixB, submatrixC, batches, 
                                   workgroupSize, NlocalCache, verbose);
  } else {
    Rcpp::warning("class of var must be fvclMatrix or dvclMatrix");
    result = Rcpp::wrap(1L);
  }
  //#endif  
  return result;
  
}



