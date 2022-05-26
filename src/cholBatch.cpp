#include "lgmlikFit.hpp"
//#define DEBUG

// Nlocal[0] is Drow, Nlocal[1] inner loop over columns, 
// Nglobal[0] is Dmatrix
// local and global work sizes should be identical for dimension 1 (second dimension), only 1 group for dimension 1


template <typename T> std::string cholBatchKernelString(
    int colStart,
    int colEnd,
    int N,
    int Npad,
    int NpadDiag,
    int NpadBetweenMatrices,
    int NstartA,
    int NstartD,
    Rcpp::IntegerVector Ncache, 
    Rcpp::IntegerVector Nlocal, // length 2
    int allowOverflow,
    int logDet) {
  
  std::string typeString = openclTypeString<T>();
  std::string result = "";
  
  
  if(typeString == "double") {
    result += "\n#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\n";
  }
  
  result +=
    "\n#define N " + std::to_string(N) + "  //dimension of matrix\n"
    "#define colStart " + std::to_string(colStart) + "  //column to start at\n"
    "#define colEnd " + std::to_string(colEnd) + "\n"
    "//internal number of columns\n#define Npad " + std::to_string(Npad) + "\n"
    "//internal columns for matrix holding diagonals\n#define NpadDiag " + std::to_string(NpadDiag) + "\n"
    "#define NstartA " + std::to_string(NstartA) + "\n"
    "#define NstartD " + std::to_string(NstartD) + "\n"
    "//elements in internal cache\n#define Ncache " + std::to_string(Ncache[0]) + "\n"
    "//extra rows between stacked matrices\n#define NpadBetweenMatrices " + std::to_string(NpadBetweenMatrices) + "\n"
    "#define maxLocalItems " + std::to_string(Nlocal[0]*Nlocal[1]) + "\n";
  
  
  result += "\n__kernel void cholBatch(\n"
  "	__global " + typeString + " *A,\n" 
  "	__global " + typeString + " *diag,\n"
  " __local " + typeString + " *diagLocal,\n"
  "              int Nmatrix";
  
  if(logDet){
    result += 
      ",\n	__global " + typeString + " *logDet,\n"
      "                int logDetIndex\n";
  }
  
  
  result += "\n){\n"
  " const int localIndex = get_local_id(0)*get_local_size(1) + get_local_id(1);\n"
  " const int localIndexIsZero = (localIndex==0);\n"
  " const int NlocalTotal = get_local_size(0)*get_local_size(1);\n"
  
//  " local " + typeString + " diagLocal[Ncache];//local cache of diagonals\n"
  
  " local " + typeString + " toAddLocal[maxLocalItems];\n"
  " int Dcol, DcolNpad;\n"
  " int Drow, DrowBlock, Dk, Dmatrix, DmatrixBlock;\n";
  
  if(allowOverflow) {
    result += "int minDcolNcache;\n";
  }
  
  result += " " +  typeString + " DL;\n" 
  " local " +  typeString + " diagDcol;\n" 
  " int AHere, AHereDcol, AHereDrow, diagHere;\n";
  
  if(logDet){
    result += typeString  + " logDetHere;\n";
  }
  result +=   " barrier(CLK_LOCAL_MEM_FENCE);\n\n";
  result +=  
    "for(Dmatrix = get_group_id(0);\n "
    "    Dmatrix < Nmatrix;\n "
    "    Dmatrix+= get_num_groups(0)){\n\n"
    
    // "for(DmatrixBlock=0, Dmatrix = get_group_id(0);\n"
    // "     DmatrixBlock < Nmatrix;\n"
    // "     DmatrixBlock+= get_num_groups(0), Dmatrix+= get_num_groups(0)){\n"
    
    " diagHere = Dmatrix*NpadDiag + NstartD;\n"
    " AHere =  Dmatrix*NpadBetweenMatrices + NstartA;\n";
  if(logDet){
    result += " logDetHere = 0.0;\n";
  }
  result +=
    "\n for(Dcol = colStart; Dcol < colEnd; Dcol++) {\n"
    "  DcolNpad = Dcol*Npad;\n"
    "  AHereDcol = AHere+DcolNpad;\n"
    //"  Dcolm1 = Dcol - 1;\n"
    "  DL = 0.0;\n"
    "  diagLocal[localIndex]=0.0;\n"
    "  toAddLocal[localIndex]=0.0;\n";
  
  
  if(allowOverflow) {
    result +="  minDcolNcache = min(Dcol, Ncache);\n"
    "  for(Dk = localIndex; Dk < minDcolNcache; Dk += NlocalTotal) {\n";
  } else {
    result +="  for(Dk = localIndex; Dk < Dcol; Dk += NlocalTotal) {\n";
  }
  
  
  
  result += 
    "\n // diagonals\n"
    "    DL = A[AHereDcol+Dk];\n"
    "    diagLocal[Dk] = diag[diagHere+Dk] * DL;\n"// cached A[Dcol, 1:Dcol] D[1:Dcol]
    "    toAddLocal[localIndex] += diagLocal[Dk] * DL;\n"
    "  }// Dk\n"; 
  
  
  if(allowOverflow) {
    result +=
      "  for(Dk=minDcolNcache+localIndex; Dk < Dcol; Dk += NlocalTotal) {\n"
      "    DL = A[AHereDcol+Dk];\n"
      "    toAddLocal[localIndex] += diag[diagHere+Dk] * DL * DL;\n"
      "  }// Dk\n";
  }
  
  result +=  
    "\n// reduction on dimension 1\n";
  
  result += "  barrier(CLK_LOCAL_MEM_FENCE);\n"
  "  if( (get_local_id(1) == 0)){\n"
  "   for(Dk = 1; Dk < get_local_size(1); Dk++) {\n"
  "    toAddLocal[localIndex] +=  toAddLocal[localIndex + Dk];\n"
  "   }//for Dk\n"
  "  }//get_local_id(1) == 0\n"
  "  barrier(CLK_LOCAL_MEM_FENCE);\n\n";
  
  result +=     
    "// final reduction on dimension 0\n"
    "  if(localIndexIsZero ){\n";
  
  result +=     
    "   for(Dk = get_local_size(1); Dk < NlocalTotal; Dk+= get_local_size(1)) {\n"   //   localIndex + get_local_size(1)
    "    toAddLocal[localIndex] +=  toAddLocal[Dk];\n"
    "   } //Dk\n";
  
  result +=     
    "   diagDcol = A[AHereDcol+Dcol] - toAddLocal[localIndex];\n"
    "   diag[diagHere+Dcol] = diagDcol;\n";
  
  if(logDet){
    result += "  logDetHere += log(diagDcol);\n";
  }
  
  result +=  
    "#ifdef diagToOne\n"
    "   A[AHereDcol+Dcol] = 1.0;\n"
    "#endif\n"
    "  }  //localIndex==0 and Dmatrix < Nmatrix\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n\n";
  //"  diagDcol = diagHere[Dcol];\n"
  
  result += "// off diagonals\n";
  //  result += " for(Drow = Dcol+get_local_id(0)+1; Drow < N; Drow += get_local_size(0)) {\n";
  
  // DrowBlock is the Drow for the first work item.  
  // sometimes DrowBlock < N but Drow > N
  // need if statement to stop those rows from being computed
  // can't have condition Drow < N in loop
  // because some work items won't do the loop and 
  // memory fence won't work
  
  result += " for(DrowBlock = Dcol+1; DrowBlock < N; DrowBlock += get_local_size(0)) {\n";
  
  result += "  Drow = DrowBlock + get_local_id(0);\n";
  
  result +=
    "  AHereDrow = AHere+Drow*Npad;\n"
    "  DL = 0.0;\n";
  
  result += "  if(Drow < N){\n";
  
  if(allowOverflow) {
    result +=
      "  for(Dk = get_local_id(1); Dk < minDcolNcache; Dk+=get_local_size(1)) {\n";
  } else {
    result +=
      "  for(Dk = get_local_id(1); Dk < Dcol; Dk+=get_local_size(1)) {\n";
  }
  
  result +=
    "   DL += A[AHereDrow+Dk] * diagLocal[Dk];\n"
    // DL -= A[Drow + Dk * Npad] * A[Dcol + DkNpad] * diag[Dk];"
    "  } // Dk\n";
  
  
  
  if(allowOverflow) {
    result +=
      "	 for(minDcolNcache + get_local_id(1); Dk < Dcol; Dk+=get_local_size(1)) {\n"
      "    DL += A[AHereDrow+Dk] * diag[diagHere+Dk] * A[AHereDcol+Dk];\n"
      "  }// Dk\n"; 
  }
  
  
  result += "   }//Drow < N\n";
  
  result +=
    "  toAddLocal[localIndex] = DL;\n\n";
  
  
  result +=
    "  // local reduction\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n";
  result +=
    "  if( (get_local_id(1) == 0) & (Drow < N)){\n"
    "   DL = toAddLocal[localIndex];\n";
  
  result +=    "   for(Dk = 1; Dk < get_local_size(1); Dk++) {\n"
  "    DL +=  toAddLocal[localIndex + Dk];\n"
  "   }//Dk\n"; 
  
  result +=    
    "   A[AHereDrow+Dcol] = (A[AHereDrow+Dcol] - DL)/diagDcol;\n"
    "  }//get_local_id(1) == 0\n" 
    "  barrier(CLK_GLOBAL_MEM_FENCE);\n\n";
  
  result +=
    " }//DrowBlock\n";
  
  result +=  
    "barrier(CLK_GLOBAL_MEM_FENCE);\n"
    "} // Dcol loop\n\n";
  
  if(logDet){
    result +=  "if(localIndexIsZero){\n"   //& (Drow < N)
    " logDet[logDetIndex + Dmatrix] = logDetHere;\n"
    "}\n";
  }
  
  result += "barrier(CLK_GLOBAL_MEM_FENCE);\n"
  "} // Dmatrix loop\n\n";
  
  result +=   "barrier(CLK_LOCAL_MEM_FENCE);\n"
  "}// kernel\n";
  return(result);
}







/*
 const std::vector<int> &Nglobal,
 const std::vector<int> &Nlocal, 
 const std::vector<int> &NlocalCache,
 */




template <typename T> 
int cholBatchVcl(
    viennacl::matrix<T> &A,
    viennacl::matrix<T> &D,
    Rcpp::IntegerVector Astartend, // submatrices, matrix A[Astartend[1]:Astartend[2]:Astartend[3]:Astartend[4]]
    Rcpp::IntegerVector Dstartend,  
    const int numbatchD,
    Rcpp::IntegerVector Nglobal,
    Rcpp::IntegerVector Nlocal, 
    Rcpp::IntegerVector NlocalCache,
    const int ctx_id) {
  
  const int NstartA = A.internal_size2() * Astartend[0] + Astartend[2];
  const int NstartD = D.internal_size2() * Dstartend[0] + Dstartend[2];
  
  viennacl::ocl::local_mem localCache( NlocalCache[0]*sizeof(A(0,0)) );
  
  std::string cholClString = cholBatchKernelString<T>(
    0L, // start
    Astartend[3], // colEnd  
             Astartend[3], // N
                      A.internal_size2(), // Npad
                      D.internal_size2(),
                      A.size2() * A.internal_size2(),// NpadBetweenMatrices,
                      NstartA,
                      NstartD,
                      NlocalCache, 
                      Nlocal,
                      ((int) A.size2() ) > NlocalCache[0], // allow overflow  // needs change?
                                                      0L // logDet
  );
  
  viennacl::ocl::context ctx(viennacl::ocl::get_context(ctx_id));
  viennacl::ocl::program & my_prog = ctx.add_program(cholClString, "my_kernel");
  
#ifdef DEBUG
  
  Rcpp::Rcout << cholClString << "\n\n";
  
#endif  
  
  viennacl::ocl::kernel & cholKernel = my_prog.get_kernel("cholBatch");
  
  if(Nlocal[1] != Nglobal[1]) {
    Rcpp::warning("local and global work sizes should be identical for dimension 2, ignoring global");
  }
  
  // dimension 0 is cell, dimension 1 is matrix
  cholKernel.global_work_size(0, (cl_uint) (Nglobal[0] ) );
  cholKernel.global_work_size(1, (cl_uint) (Nlocal[1] ) );
  
  cholKernel.local_work_size(0, (cl_uint) (Nlocal[0]));
  cholKernel.local_work_size(1, (cl_uint) (Nlocal[1]));
  
  viennacl::ocl::command_queue theQueue = cholKernel.context().get_queue();
  viennacl::ocl::enqueue(cholKernel(A, D, localCache, numbatchD), theQueue);  
  clFinish(theQueue.handle().get());
  //viennacl::ocl::enqueue(cholKernel(A, D, numbatchD));
  return 0L;
}













template<typename T> 
void cholBatchTemplated(
    Rcpp::S4 A,
    Rcpp::S4 D,
    Rcpp::IntegerVector Astartend,
    Rcpp::IntegerVector Dstartend,
    const int numbatchD,
    Rcpp::IntegerVector Nglobal,
    Rcpp::IntegerVector Nlocal, 
    Rcpp::IntegerVector NlocalCache) {
  
  const bool BisVCL=1;
  const int ctx_id = INTEGER(A.slot(".context_index"))[0]-1;
  std::shared_ptr<viennacl::matrix<T> > vclA = getVCLptr<T>(A.slot("address"), BisVCL, ctx_id);
  std::shared_ptr<viennacl::matrix<T> > vclD = getVCLptr<T>(D.slot("address"), BisVCL, ctx_id);
  
  cholBatchVcl<T>(
    *vclA, *vclD, 
    Astartend, Dstartend,
    numbatchD,
    Nglobal, 
    Nlocal,
    NlocalCache,
    ctx_id);
  
}















//[[Rcpp::export]]
void cholBatchBackend(
    Rcpp::S4 A,
    Rcpp::S4 D,
    Rcpp::IntegerVector Astartend,
    Rcpp::IntegerVector Dstartend,
    const int numbatchD,
    Rcpp::IntegerVector Nglobal,
    Rcpp::IntegerVector Nlocal,
    Rcpp::IntegerVector NlocalCache) {
  
  
  Rcpp::traits::input_parameter< std::string >::type classVarR(RCPP_GET_CLASS(A));
  std::string precision_type = (std::string) classVarR;
  
  if(precision_type == "fvclMatrix") {
    cholBatchTemplated<float>(
      A, D, Astartend, Dstartend,
      numbatchD, Nglobal, Nlocal,
      NlocalCache);
  } else if (precision_type == "dvclMatrix") {
    cholBatchTemplated<double>(
      A, D, Astartend, Dstartend,
      numbatchD, Nglobal, Nlocal,
      NlocalCache);
  } else {
    Rcpp::warning("class of A must be fvclMatrix or dvclMatrix");
  }
}
