#include "lgmlikFit.hpp"
#define DEBUG
//#define NOKERNELS

// C = A^T A or A^T D A or A^T D^(-1) A 
// if onlyDiagC = 1, then only compute diagonals of C
// C is an Nmatrix by Ncol matrix


template <typename T> 
std::string crossprodBatchString(
    const int Nrow,    // Nrow of A single batch
    const int Ncol,     // ncol of A single batch
    //    const int Nmatrix, 
    const int NpadC, // ignored if onlyDiagC, use NpadBetweenMatrices in this case
    const int NpadA,
    const int NpadD, // set to zero to omit D
    const int invertD, // set to 1 for A^T D^(-1) A
    const int onlyDiagC, // set to 1 to only compute diagonals of C
    const int NstartC,  // newly added
    const int NstartA,  // new
    const int NstartD,  // new
    const int NpadBetweenMatricesC,
    const int NpadBetweenMatricesA,
    const int NlocalCacheD // numbers of rows to cache of A  Rcpp::IntegerVector Nlocal// cache a Nlocal[0] by Nlocal[1] submatrix of C
) { 
  
  /*
   * global groups dimension 1 is Dcol, groups dimension 2 is Dmatrix
   * local items dimension 1 is Dinner, dimension 2 is Drow
   * 
   */
  
  std::string typeString = openclTypeString<T>();
  std::string result = "";
  const int NrowStop = std::min(NlocalCacheD, Nrow);
  
  if(typeString == "double") {
    result += "\n#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\n";
  }
  result += 
    "\n#define Nrow " + std::to_string(Nrow) + "\n"    
    "#define Ncol " + std::to_string(Ncol) + "\n"  
    //    "#define Nmatrix " + std::to_string(Nmatrix) + "\n"    
    "#define NpadC " + std::to_string(NpadC) + "\n"    
    "#define NpadA " + std::to_string(NpadA) + "\n"    
    "#define NpadD " + std::to_string(NpadD) + "\n"   
    "#define NstartC " + std::to_string(NstartC) + "\n"   
    "#define NstartA " + std::to_string(NstartA) + "\n" 
    "#define NstartD " + std::to_string(NstartD) + "\n" 
    //    "#define NpadLocal " + std::to_string(Nlocal[1]) + "\n"    
    "#define NpadBetweenMatricesC " + std::to_string(NpadBetweenMatricesC) + "\n"    
    "#define NpadBetweenMatricesA " + std::to_string(NpadBetweenMatricesA) + "\n"    
    "#define NrowStop " + std::to_string(NrowStop) + "\n"    
    "#define NlocalCacheD "  + std::to_string(NlocalCacheD) + "\n\n";   
  
  result +=
    "__kernel void crossprodBatch(\n"
    "global " + typeString+ " *C,\n"
    "const global "+ typeString+ " *A,\n";
  
  if (NpadD >0 ) {
    result +=  "const global " + typeString + " *D,\n";
  }
  result +=
    " __local " + typeString + " *localCache,\n"
    " int NrowStartC,\n"
    " int Nmatrix) {\n\n"
    
    "local " + typeString + " *Acache=localCache;\n" 
    "local " + typeString + " *Dcache = &localCache[" + 
      std::to_string(NlocalCacheD) + "];\n" 
  "local " + typeString + " *Ccache = &localCache[" + 
    std::to_string(2*NlocalCacheD) + "];\n" +
    
    typeString + " Cout, Ctemp;\n"
  "event_t ev;\n"
  "const int NpadLocal = get_local_size(1);\n"
  "int AHere, CHere;\n"
  "int Dmatrix, Drow, Dcol, DrowNpadC, Dinner, DinnerA, DinnerAcol, DrowBlock, DinnerBlock, DcolBlock, DcolInBounds, DrowInBounds;\n"
  "int A0Dcol, A0Drow;// location of elements A[0,Dcol] and A[0,Drow]\n"
  "const int AHereInc = get_num_groups(1)*NpadBetweenMatricesA;\n";
  
  
  if(onlyDiagC) {
    result +=
      "const int CHereInc = get_num_groups(1)*NpadC;\n"; // made changes here
  }else{
    result +=
      "const int CHereInc = get_num_groups(1)*NpadBetweenMatricesC;\n"; // made changes here
  }
  
  
  result +=
    //"const int DrowNpadCInc = get_local_size(1)*NpadC;\n"
    "const int localIndex = get_local_id(0) * get_local_size(1) + get_local_id(1);\n"
    "const int NlocalTotal = get_local_size(1)*get_local_size(0);\n"
    "const int cacheIndex = get_local_id(1)+NpadLocal*get_local_id(0);\n";
  
  if(onlyDiagC) {
    result +=    "const int doLocalSum = (localIndex==0);\n";
    "const int DinnerAinc = NlocalTotal*NpadA;\n";
  } else {
    result +=    "const int doLocalSum = (get_local_id(0)==0);\n";
    "const int DinnerAinc = get_local_size(0)*NpadA;\n";
  }
  
  
  if(NpadD) {
    result += "int DHere;\n"
    "const int DHereInc = get_num_groups(1)*NpadD;\n";
  }
  
  
  result +=  "\n\n"
  "for(Dmatrix = get_group_id(1),\n"
  "    AHere = Dmatrix * NpadBetweenMatricesA + NstartA,\n"; // made changes here
  if(NpadD) {
    result +=
      "    DHere = Dmatrix * NpadD + NstartD,\n"; // made changes here
  }
  
  if(onlyDiagC) {
    result +=
      "    CHere = (Dmatrix + NrowStartC) * NpadC + NstartC;\n"; // made changes here
  }else{
    result +=
      "    CHere = (Dmatrix + NrowStartC) * NpadBetweenMatricesC + NstartC;\n"; // made changes here
  }
  
  result +=
    "     Dmatrix < Nmatrix;\n"
    "     Dmatrix += get_num_groups(1),\n"
    "     AHere += AHereInc,\n";
  
  if(NpadD) {
    result += "    DHere += DHereInc,\n";
  }
  result +=
    "    CHere += CHereInc\n"
    "  ){\n";
  
  // result +=  "\n\n"
  // "   for(Dmatrix = get_group_id(1);\n"
  // "       Dmatrix < Nmatrix;\n"
  // "       Dmatrix += get_num_groups(1)\n"
  // "    ){\n";
  // 
  // result += 
  //   "     AHere = Dmatrix * NpadBetweenMatricesA + NstartA;\n";
  // if(NpadD) {
  //   result +=
  //     "     DHere = Dmatrix * NpadD + NstartD,\n"; 
  // }
  // if(onlyDiagC) {
  //   result +=
  //     "     CHere = (Dmatrix + NrowStartC) * NpadC + NstartC;\n"; 
  // }else{
  //   result +=  
  //     "     CHere = (Dmatrix + NrowStartC) * NpadBetweenMatricesC + NstartC;\n"; 
  // }
  
  
  if(NpadD && NrowStop) {
    
    result +=  "\n"
    "  /* cache first NrowStop elements of D for this matrix*/\n"
    "   ev=async_work_group_copy(\n"
    "     Dcache, &D[DHere],\n"
    "     NrowStop, 0);\n"
    "   wait_group_events (1, &ev);\n"
    "   barrier(CLK_LOCAL_MEM_FENCE);\n";      // barrier needed
    
  }
  
  result +=  "\n"
  "  for(Dcol = get_group_id(0);\n"
  "      Dcol < Ncol;\n"
  "      Dcol += get_num_groups(0)\n"
  "      ){\n\n";
  
  result += 
    "      A0Dcol = AHere + Dcol;\n";
  
  
  
  if(NrowStop) {
    
    result +=  "\n"
    "     // cache A[1:NrowStop, Dcol]\n"
    "     ev=async_work_group_strided_copy (\n"
    "       Acache, &A[A0Dcol],\n"
    "       NrowStop, NpadA, 0);\n"
    "     wait_group_events (1, &ev);\n"
    "     barrier(CLK_LOCAL_MEM_FENCE);\n\n";    // barrier needed
    
    if(onlyDiagC) {
      result +=
        // "    // square the cached A\n"
        // "    for(Dinner = localIndex;\n"
        // "        Dinner < NrowStop;\n"
        // "        Dinner+=NlocalTotal\n"
        // "    ){\n";   
        "    // square the cached A\n"
        "    for(DinnerBlock = 0;\n"
        "        DinnerBlock < NrowStop;\n"
        "        DinnerBlock+=NlocalTotal\n"
        "    ){\n"
        "      Dinner = DinnerBlock + localIndex;\n";   
      result +=
        "      if(Dinner < NrowStop) Acache[Dinner] *= Acache[Dinner];\n";
      result +=
        "     }//Dinner\n"
        "     barrier(CLK_LOCAL_MEM_FENCE);\n\n";
    }
    if(NpadD) {
      if(onlyDiagC) {
        result +=
          "    // multiply by D\n"
          "    for(DinnerBlock = 0;\n"
          "        DinnerBlock < NrowStop;\n"
          "        DinnerBlock += NlocalTotal\n"
          "    ){\n"
          "      Dinner = DinnerBlock + localIndex;\n  "
          "      if(Dinner < NrowStop) {\n";
        if(invertD) {
          result +=
            "      Acache[Dinner] /= Dcache[Dinner];\n";
        } else {
          result +=
            "      Acache[Dinner] *= Dcache[Dinner];\n";
        }
        result +=
          "      }\n"
          "    }//Dinner\n"
          "    barrier(CLK_LOCAL_MEM_FENCE);\n\n";
      }else{
        result +=
          // "    // multiply by D\n"
          // "    for(Dinner = get_local_id(0);\n"
          // "        Dinner < NrowStop;\n"
          // "        Dinner+=get_local_size(0)\n"
          // "    ){\n";   
          "    // multiply by D\n"
          "    for(DinnerBlock = 0;\n"
          "        DinnerBlock < NrowStop;\n"
          "        DinnerBlock += get_local_size(0)\n"
          "    ){\n"
          "         Dinner = DinnerBlock + get_local_id(0);\n  "
          "         if( Dinner < NrowStop )\n";
        if(invertD) {
          result +=
            "         Acache[Dinner] /= Dcache[Dinner];\n";
        } else {
          result +=
            "      Acache[Dinner] *= Dcache[Dinner];\n";
        }
        result +=
          "    }//Dinner\n"
          "    barrier(CLK_LOCAL_MEM_FENCE);\n\n";
      } // else
    } // NpadD
  } // NrowStop
  
  
  
  if(onlyDiagC) {
    result +=
      "      Drow = Dcol;\n";
    // result += 
    //   "     Drow = Dcol;\n"
    //   "     A0Drow = AHere + Drow;\n"
    //   "     DrowNpadC = CHere + Drow * NpadC;\n\n";
  }  else {
    
    result +=
      "    for(DrowBlock = Dcol;\n"
      "        DrowBlock < Ncol;\n "
      "        DrowBlock += get_local_size(1)) {\n"
      "      Drow = DrowBlock + get_local_id(1);\n";
    
    // result += 
    //   "   for(Drow = Dcol + get_local_id(1),\n"
    //   "       DrowNpadC = CHere + Drow * NpadC,\n"
    //   "       A0Drow = AHere + Drow;\n"
    //   "     Drow < Ncol;\n"
    //   "     Drow += get_local_size(1),\n" 
    //   "       DrowNpadC += DrowNpadCInc,\n"
    //   "       A0Drow +=  get_local_size(1)\n"
    //   "   ){\n\n";
        }
   
   result +=
    "      DrowInBounds = Drow < Ncol;\n"
    "      DrowNpadC = CHere + Drow * NpadC;\n"
    "      A0Drow = AHere + Drow;\n"
    //"      barrier(CLK_LOCAL_MEM_FENCE);\n"
    "      Cout=0.0;\n\n";
  
  // result += 
  //   "      Cout=0.0;\n\n";


  // if(onlyDiagC) {
  //   result +=
  //     "      // cached parts\n"
  //     "     for(Dinner = localIndex;\n"
  //     "         Dinner < NrowStop;\n"
  //     "         Dinner += NlocalTotal\n"
  //     "      ){\n";
  // }else {
  //   result +=
  //     "      // cached parts\n"
  //     "     for(Dinner =  get_local_id(0);\n"
  //     "         Dinner < NrowStop;\n"
  //     "         Dinner += get_local_size(0)\n"
  //     "      ){\n";
  // }
  // 
  // result += "         DinnerA = A0Drow + Dinner*NpadA;\n";
  // 
  // result +=
  //   "         if(DrowInBounds) {\n";
  // 
  // if(onlyDiagC) {    
  //   result += 
  //     "           Cout += Acache[Dinner];\n"; 
  // } else {
  //   result += 
  //     "          Cout += A[DinnerA] * Acache[Dinner];\n";
  //   //    "          Cout += A[Dmatrix * NpadBetweenMatricesA + Dinner*NpadA + Drow] * Acache[Dinner];\n";
  // }
  // result +=     
  //   "        } // if in bounds \n"
  //   "      } // Dinner\n\n";    
  // 
  
  
  result +=
    "      // cached parts\n"
    "     for(DinnerBlock = 0;\n"
    "         DinnerBlock < NrowStop;\n";
  if(onlyDiagC) {
    result +=
      "         DinnerBlock += NlocalTotal\n"
      "      ){\n"
      "         Dinner = DinnerBlock + localIndex;\n";
  } else {
    result +=
      "         DinnerBlock += get_local_size(0)\n"
      "      ){\n"
      "         Dinner = DinnerBlock + get_local_id(0);\n";
  }

  result += "         DinnerA = A0Drow + Dinner*NpadA;\n";

  result +=
    "         if(DrowInBounds & (Dinner < NrowStop) ) {\n";

  if(onlyDiagC) {
    result +=
      "           Cout += Acache[Dinner];\n";
  } else {
    result +=
      "          Cout += A[DinnerA] * Acache[Dinner];\n";
    //    "          Cout += A[Dmatrix * NpadBetweenMatricesA + Dinner*NpadA + Drow] * Acache[Dinner];\n";
  }
  result +=
    "        } // if in bounds \n"
    "      } // DinnerBlock\n\n";
   // "      barrier(CLK_LOCAL_MEM_FENCE);\n\n\n";

  result +=
    "       // un-cached parts\n"
    "       for(DinnerBlock = NrowStop;\n"
    "           DinnerBlock < Nrow;\n";
  
  if(onlyDiagC) {
    result +=
      "         DinnerBlock += NlocalTotal\n"
      "      ){\n"
      "         Dinner = DinnerBlock + localIndex;\n";
  }else{
    result +=
      "         DinnerBlock += get_local_size(0)\n"
      "      ){\n"
      "         Dinner = DinnerBlock + get_local_id(0);\n";
  }
  
  result += 
    "           DinnerA = A0Drow + Dinner*NpadA;\n"
    "           DinnerAcol = A0Dcol + Dinner*NpadA;\n";
  
  result +=
    "        if(DrowInBounds & (Dinner < Nrow) ){\n";
  if(NpadD) {
    if(invertD) {
      result += 
        "      Cout += A[DinnerA] * A[DinnerAcol] / D[DHere+Dinner];\n";
    } else {
      result += 
        "      Cout += A[DinnerA] * A[DinnerAcol] * D[DHere+Dinner];\n";
    }
  } else {
    result += 
      "      Cout += A[DinnerA] * A[DinnerAcol];\n";
  }  
  result+= 
    "       }// if in bounds \n"
    "       }// DinnerBlock\n\n";
  
  result +=       
    "      Ccache[cacheIndex] = Cout;\n"
    "      barrier(CLK_LOCAL_MEM_FENCE);\n\n\n";    
  
  result +=
    "      if(doLocalSum && DrowInBounds){\n";
  if(onlyDiagC) {
    result +=
      "        for(Dinner = 1;Dinner < NlocalTotal;Dinner++){\n"
      "          Ccache[cacheIndex] += Ccache[cacheIndex + Dinner];\n"
      "        }\n";
    result += "          C[CHere + Dcol] = Ccache[cacheIndex];\n";
    // DEBUG
    //        result += "          C[CHere + Dcol] = CHere + Dcol;\n";
  } else {
    result +=
      "        for(Dinner = 1;Dinner < get_local_size(0);Dinner++){\n"
      "          Ccache[cacheIndex] += Ccache[cacheIndex + Dinner * NpadLocal];\n"
      "        }\n";
    
    result += 
      "          C[DrowNpadC + Dcol] = Ccache[cacheIndex];\n";
  } 
  
  result +=
    "      }//doLocalSum \n"
    "      barrier(CLK_LOCAL_MEM_FENCE);\n\n";    //  barrier here!
  
  
  if(!onlyDiagC) {
    result += 
      "    }// Drow\n"
      "  barrier(CLK_LOCAL_MEM_FENCE);\n\n";
  }
  result += 
    "  }// Dcol\n";
  
  result += 
    //  "  barrier(CLK_LOCAL_MEM_FENCE);\n" 
    "  }// Dmatrix\n";
    //"  barrier(CLK_LOCAL_MEM_FENCE);\n";
  result += 
    "}";
  
  return(result);
}

/*
 std::vector<int> Nglobal,
 std::vector<int> Nlocal,*/





template <typename T> 
int crossprodBatch(
    viennacl::matrix<T> &C,  // must be a batch of square matrices 
    viennacl::matrix<T> &A,
    viennacl::matrix<T> &D,
    const int invertD,
    Rcpp::IntegerVector Cstartend,
    Rcpp::IntegerVector Astartend,
    Rcpp::IntegerVector Dstartend,  
    Rcpp::IntegerVector Nglobal,
    Rcpp::IntegerVector Nlocal, 
    const int NlocalCache, // rows cached are (NlocalCache - prod(Nlcoal))/2
    const int ctx_id,
    const int verbose) {
  
  const int Ncol = Astartend[3];
  const int Nmatrix = C.size1()/C.size2();
  const int Nrow = Astartend[1];
  
  const int NstartC = C.internal_size2() * Cstartend[0] + Cstartend[2];
  const int NstartA = A.internal_size2() * Astartend[0] + Astartend[2];
  const int NstartD = D.internal_size2() * Dstartend[0] + Dstartend[2];
  
  const int NlocalCacheD = std::max( 0, (NlocalCache - Nlocal[0]*Nlocal[1])/2 );
  
  
  // the context
  viennacl::ocl::context ctx(viennacl::ocl::get_context(ctx_id));
  
  viennacl::ocl::local_mem localCache(NlocalCache*sizeof(A(0,0) ) );
  
  //  cl_device_type type_check = ctx.current_device().type();
  
  /* const int NpadC, 
   const int NpadA,
   const int NpadD, // set to zero to omit D
   const int invertD, // set to 1 for A^T D^(-1) A
   const int NpadBetweenMatricesC,
   const int NpadBetweenMatricesA,
   */
  std::string clString =  crossprodBatchString<T>(  
    Nrow, 
    Ncol, // ncol
    //    Nmatrix,
    C.internal_size2(), 
    A.internal_size2(), 
    D.internal_size2(),
    invertD, // A^T D^(-1) A
    0, // don't only compute diagonals of C,  onlyDiagC, // set to 1 to only compute diagonals of C
    NstartC,
    NstartA,
    NstartD,
    C.internal_size2()*C.size2(),//NpadBetweenMatricesC,
    A.internal_size2()*A.size1()/Nmatrix,//NpadBetweenMatricesA,
    NlocalCacheD  
    // Nlocal
  );
  
  
  if(verbose)  
    std::cout << "\n NlocalCacheD " << NlocalCacheD << "\n";
  
  if(verbose > 1)
    Rcpp::Rcout << clString << "\n\n";
  
  
  
  viennacl::ocl::program & my_prog = ctx.add_program(clString, "my_kernel");
#ifndef NOKERNELS  
  viennacl::ocl::kernel & multiplyKernel = my_prog.get_kernel("crossprodBatch");
  
  multiplyKernel.global_work_size(0, Nglobal[0]);
  multiplyKernel.global_work_size(1, Nglobal[1]);
  
  multiplyKernel.local_work_size(0, Nlocal[0]);
  multiplyKernel.local_work_size(1, Nlocal[1]);
  
  // diagonals and diagTimesRowOfA
  viennacl::ocl::enqueue(multiplyKernel( C, A, D, localCache, 0, Nmatrix));
#endif  
  return 0L;
}



template <typename T> 
SEXP crossprodBatchTyped(
    Rcpp::S4 CR,
    Rcpp::S4 AR,
    Rcpp::S4 DR,
    const int invertD,
    Rcpp::IntegerVector Cstartend,
    Rcpp::IntegerVector Astartend,
    Rcpp::IntegerVector Dstartend,  
    Rcpp::IntegerVector Nglobal,
    Rcpp::IntegerVector Nlocal, 
    const int NlocalCache,
    const int verbose//, int NrowStartC
) {
  /*
   std::vector<int> Nglobal = Rcpp::as<std::vector<int> >(NglobalR);
   std::vector<int> Nlocal = Rcpp::as<std::vector<int> >(NlocalR);*/
  
  const int ctx_id = INTEGER(CR.slot(".context_index"))[0]-1;
  const bool BisVCL=1;
  
  
  
  std::shared_ptr<viennacl::matrix<T> > 
    AG = getVCLptr<T>(AR.slot("address"), BisVCL, ctx_id);
  std::shared_ptr<viennacl::matrix<T> > 
    CG = getVCLptr<T>(CR.slot("address"), BisVCL, ctx_id);
  std::shared_ptr<viennacl::matrix<T> > 
    DG = getVCLptr<T>(DR.slot("address"), BisVCL, ctx_id);
  
  
  
  return Rcpp::wrap(crossprodBatch<T>(*CG, *AG, *DG, invertD, Cstartend, Astartend, Dstartend, Nglobal, Nlocal, NlocalCache, ctx_id, verbose));
  
}


//' Multiply crossproduct matrices
//' 
//' Computes C = t(A) D A
// [[Rcpp::export]]
SEXP crossprodBatchBackend(
    Rcpp::S4 C,
    Rcpp::S4 A,
    Rcpp::S4 D,
    const int invertD,
    Rcpp::IntegerVector Cstartend,
    Rcpp::IntegerVector Astartend,
    Rcpp::IntegerVector Dstartend, 
    Rcpp::IntegerVector Nglobal,
    Rcpp::IntegerVector Nlocal, 
    const int NlocalCache,
    const int verbose//, int NrowStartC
) {
  
  SEXP result;
  
  Rcpp::traits::input_parameter< std::string >::type classVarR(RCPP_GET_CLASS(C));
  std::string precision_type = (std::string) classVarR;
  
  
  if(precision_type == "fvclMatrix") {
    result = crossprodBatchTyped<float>(C, A, D, invertD,  Cstartend, Astartend, Dstartend, Nglobal, Nlocal, NlocalCache, verbose);
  } else if (precision_type == "dvclMatrix") {
    result = crossprodBatchTyped<double>(C, A, D, invertD, Cstartend, Astartend, Dstartend, Nglobal, Nlocal, NlocalCache, verbose);
  } else {
    result = Rcpp::wrap(1L);
  }
  return(result);
  
}













