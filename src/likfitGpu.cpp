// likfitGpu_p 9.30 10pm version
#include "lgmlikFit.hpp"


using namespace Rcpp;
using namespace viennacl; 
using namespace viennacl::linalg;

#define NlocalParams 22

// Ncol = Ncovariates + Ndatasets
//  XVYXVX[Nrowstart + 1:Nparamsets, 1:Ncol, 1:Ncol] = ssqYX[1:Nparamsets, 1:Ncol, 1:Ncol]
// something broken here
template <typename T> 
std::string extract_block_string_simple(int Ncovariates,    
                                 int Ndatasets,
                                 int NpadColXVYXVX, 
                                 int NpadColYX,
                                 int NpadBetweenMatricesXVYXVX,  // = NpadColXVYXVX * Ncovariates
                                 int NpadBetweenMatricesYX) { 
  
  std::string typeString = openclTypeString<T>();
  
  std::string result = "";
  if(typeString == "double") {
    result += "\n#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
  }
  
  
  result += 
    "#define Ncovariates " + std::to_string(Ncovariates) + "\n"
    "#define Ndatasets " + std::to_string(Ndatasets) + "\n"   
    "#define Ncol " + std::to_string(Ncovariates + Ndatasets) + "\n"
    "#define NpadColXVYXVX " + std::to_string(NpadColXVYXVX) + "\n"
    "#define NpadColYX "   + std::to_string(NpadColYX) + "\n"
    "#define NpadBetweenMatricesXVYXVX "   + std::to_string(NpadBetweenMatricesXVYXVX) + "\n"   
    "#define NpadBetweenMatricesYX "   + std::to_string(NpadBetweenMatricesYX) + "\n";

  result += 
    "\n__kernel void extract_block(\n"
    "__global " + typeString + " *XVYXVX,\n"
    "__global " + typeString + " *ssqYX,\n"  
    " int Nrowstart,\n"
    " int NthisIteration ){\n\n"; // Nparamsets
  
  result +=
    "  int Nstart = Nrowstart*NpadBetweenMatricesXVYXVX; \n"
    "  int Drow, DrowXVYXVX, DrowssqYX, Dcol, Dmatrix, DmatrixXVYXVX, DmatrixssqYX;\n"
    "  for(Dmatrix = 0; Dmatrix < NthisIteration; Dmatrix++){\n"
    "    DmatrixXVYXVX = Dmatrix*NpadBetweenMatricesXVYXVX + Nstart;\n"
    "    DmatrixssqYX = Dmatrix*NpadBetweenMatricesYX;\n"
    "    for(Drow = 0; Drow < Ncol; Drow++){\n"
    "      DrowXVYXVX = DmatrixXVYXVX + Dmatrix*NpadColXVYXVX;\n"
    "      DrowssqYX = DmatrixssqYX + Dmatrix*NpadColYX;\n"
    "      for(Dcol = 0; Dcol < Ncol; Dcol++){\n"
    "         XVYXVX[DrowXVYXVX + Dcol] =\n"
    "           ssqYX[DrowssqYX + Dcol];\n"
    "      }\n"  
    "    }\n"  
    "  }\n";  

    result +=
    "}\n";
  return(result);
}


template <typename T> 
std::string extract_block_string(int Ncovariates,    
                                 int Ndatasets,
                                 int NpadColXVYXVX, 
                                 int NpadColYX,
                                 int NpadBetweenMatricesXVYXVX,  // = NpadColXVYXVX * Ncovariates
                                 int NpadBetweenMatricesYX) { 
  
  std::string typeString = openclTypeString<T>();
  
  
  std::string result = "";
  
  if(typeString == "double") {
    result += "\n#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
  }
  
  
  result += 
    "#define Ncovariates " + std::to_string(Ncovariates) + "\n"
    "#define Ndatasets " + std::to_string(Ndatasets) + "\n"    
    "#define NpadColXVYXVX " + std::to_string(NpadColXVYXVX) + "\n"
    "#define NpadColYX "   + std::to_string(NpadColYX) + "\n"
    "#define NpadBetweenMatricesXVYXVX "   + std::to_string(NpadBetweenMatricesXVYXVX) + "\n"   
    "#define NpadBetweenMatricesYX "   + std::to_string(NpadBetweenMatricesYX) + "\n";
  
  result += 
    "\n__kernel void extract_block(\n"
    "__global " + typeString + " *XVYXVX,\n"
    "__global " + typeString + " *ssqYX,\n"  
    " int Nrowstart,\n"
    " int NthisIteration ){\n\n";
  
  result +=
    "  int Nstart = Nrowstart*NpadBetweenMatricesXVYXVX; \n"
    "  int Drow, Dcol;\n"
    "  int NparamsetThisloop, actualDrow;\n"
    "  int NrowStop = NthisIteration * Ncovariates;\n"
    "  int Ncol = Ncovariates + Ndatasets; \n\n";
  
  
  result += 
    
    "  for (Drow = get_global_id(0); Drow < NrowStop;\n"
    "        Drow += get_global_size(0)) {\n"
    "    for (Dcol = get_global_id(1); Dcol < Ncol; Dcol += get_global_size(1)) {\n" 
    "      NparamsetThisloop = ( (int) Drow) /Ncovariates;\n"
    "      actualDrow = ( (int) Drow ) % Ncovariates;\n"
    
    
    "      XVYXVX[Nstart + Drow * NpadColXVYXVX + Dcol] = \n"
    "        ssqYX[ NparamsetThisloop*NpadBetweenMatricesYX +\n"
    "              (Ndatasets+actualDrow)* NpadColYX + Dcol ];\n"
    "    }\n"
    "  }\n"
    "}\n";
  
  return(result);
}












template <typename T> 
std::string extract_some_diag_string(int Ndatasets,    // number of Y's
                                     int NpadColY, 
                                     int NpadColYX,
                                     int NpadBetweenMatricesYX) { 
  
  std::string typeString = openclTypeString<T>();
  
  
  std::string result = "";
  
  if(typeString == "double") {
    result += "\n#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
  }
  
  
  result += 
    //"#define NthisIteration " + std::to_string(NthisIteration) + "\n"
    "#define Ndatasets " + std::to_string(Ndatasets) + "\n"    
    "#define NpadColY " + std::to_string(NpadColY) + "\n"
    "#define NpadBetweenMatricesYX "   + std::to_string(NpadBetweenMatricesYX) + "\n"
    "#define NpadColYX "   + std::to_string(NpadColYX) + "\n";
  
  result += 
    "\n__kernel void extract_some_diag(\n"
    "__global " + typeString + " *ssqY,\n"
    "__global " + typeString + " *ssqYX,\n"  
    " int Nrowstart,\n"
    " int NthisIteration ){\n\n";
  
  result +=
    "int Drow, Dcol;\n";
  
  result += 
    
    "for (Drow = get_global_id(0); Drow < NthisIteration; Drow += get_global_size(0)) {\n"
    "for (Dcol = get_global_id(1); Dcol < Ndatasets; Dcol += get_global_size(1)) {\n" 
    
    "ssqY[(Nrowstart+Drow) * NpadColY+Dcol] = ssqYX[ Drow*NpadBetweenMatricesYX + Dcol* NpadColYX + Dcol ];\n"
    
    "}\n"
    "}\n"
    "}\n";
  
  return(result);
}



/*
 * sum logs of rows
 * use only one work item dimension
 * groups are rows, local items sum over colums
 * 
 */
template <typename T> 
std::string logRowSumString(int NlocalCache) {  
  
  std::string typeString = openclTypeString<T>();
  
  
  std::string result = "";
  
  if(typeString == "double") {
    result += "\n#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
  }
  result += 
    "#define NlocalCache " + std::to_string(NlocalCache) + "\n";    
  
  result += 
    "\n\n__kernel void logRowSum(\n"
    "  __global " + typeString + "* x,\n"  
    "  __global " + typeString + "* output,\n" 
    "  int Nrow, int Ncol, int Npad, int startX, int startOutput"
    "){\n\n";  
  
  result += "int Drow, Dcol, Dindex;\n";
  result += "local " + typeString + " thesum[NlocalCache];\n";
  
  result += 
    
    "  for(Drow = get_group_id(0);   Drow < Nrow;    Drow+=get_num_groups(0)){\n"
    "    thesum[get_local_id(0)] = 0.0;\n"
    "    for(Dcol = get_global_id(0),   Dindex = startX + Drow*NpadCol+Dcol;\n" 
    "        Dcol < Ncol; Dcol+=get_global_size(0), Dindex+=get_global_size(0)){\n"
    "        thesum[get_local_id(0)] += log(x[Dindex]);\n"
    "    } // end loop through columns\n"
    "  if(get_local_id(0) == 0) {\n"
    "    for(Dcol = 1; Dcol < get_local_size(0); Dcol++){\n"
    "      thesum[0] += thesum[Dcol];\n"
    "    }//for\n"
    "    output[Drow + startOutput] = thesum[0];\n"
    "  }//if\n"
    "  } // end loop through rows\n"
    "}\n";
  
  return(result);
}




template <typename T> 
std::string boxcoxKernelString(int NlocalCache, int zeroCol,
                               int Nobs, int Nboxcox,
                               int Npad,
                               int boxcoxHasZero) {
  
  std::string typeString = openclTypeString<T>();
  std::string result = "";
  
  if(typeString == "double") {
    result += "\n#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\n";
  }
  result += 
    "#define NlocalCache " + std::to_string(NlocalCache) + "\n"
    "#define DstartObs 0\n"
    "#define DstartBoxcox 0\n"
    "#define zeroCol " + std::to_string(zeroCol) + "\n"
    "#define Nobs "   + std::to_string(Nobs) + "\n"
    "#define Nboxcox "   + std::to_string(Nboxcox) + "\n"
    "#define Npad  "   + std::to_string(Npad) + "\n";
  
  result +=
    "\n__kernel void boxcox(\n"
    "__global " + typeString + " *y,\n"
    "__global " + typeString + " *param,\n"  
    "__global " + typeString + " *jacobian\n"
    "){\n\n";
  
  result +=
    "int Dboxcox, Dobs;\n"
    + typeString + " logYhere, boxcoxHere;\n"
  "local " + typeString + " sumLogY[NlocalCache];\n";
  
  if (boxcoxHasZero == 1) {
    result +=
      " if(get_global_id(1)==0 & get_group_id(0) == 0){\n"
      "  sumLogY[get_local_id(0)] = 0.0;\n"
      "  for(Dobs=get_local_id(0);Dobs < Nobs; Dobs+= get_local_size(0)){\n"
      "    logYhere = log(y[DstartObs + Npad * Dobs]);\n"
      "    sumLogY[get_local_id(0)] += logYhere;\n"
      "    if(zeroCol) y[DstartObs + zeroCol + Npad * Dobs] = logYhere;\n"
      "  }// for Dobs\n"
      "  if(get_local_id(0)==0){\n"
      "    for(Dobs=1;Dobs < get_local_size(0);Dobs++){\n"
      "     sumLogY[0] += sumLogY[Dobs];\n"
      "    }// for Dobs\n"
      "    if(get_global_id(0)==0){\n"
      "     for(Dboxcox=0;Dboxcox < Nboxcox;Dboxcox++){\n"
      // " //   jacobian = -2*(BoxCox-1)* sum(log(yXcpu[,1]))\n"
      "      jacobian[DstartBoxcox + Dboxcox] =  -2*(param[DstartBoxcox + Dboxcox]-1)*sumLogY[0];\n"
      "     }// for Dboxcox\n"
      "    }// if global0\n"
      "  }// if local0\n"
      "}// if global and group\n";
    
    result +=
      "for(Dboxcox = get_global_id(1)+1; Dboxcox < Nboxcox; Dboxcox+= get_global_size(1)){\n"   // +1
      " boxcoxHere = param[DstartBoxcox + Dboxcox];\n"
      " if(Dboxcox != zeroCol) {\n"
      "  for(Dobs=get_global_id(0);Dobs < Nobs; Dobs+= get_global_size(0)){\n"
      "//    transformed_y[ ,i] <- ((yXcpu[ ,1]^BoxCox[i]) - 1)/BoxCox[i]\n"
      "      y[DstartObs + Dboxcox + Npad * Dobs] = \n"
      "        (pow(y[DstartObs + Npad * Dobs],boxcoxHere)-1)/boxcoxHere;\n"
      "    }// for Dobs\n"
      "  }// if zerocol\n"
      "}// for Dboxcox\n";
  }else if (boxcoxHasZero == 0) {
    result +=
      " if(get_global_id(1)==0 & get_group_id(0) == 0){\n"
      "  sumLogY[get_local_id(0)] = 0.0;\n"
      "  for(Dobs=get_local_id(0);Dobs < Nobs; Dobs+= get_local_size(0)){\n"
      "    logYhere = log(y[DstartObs + Npad * Dobs]);\n"
      "    sumLogY[get_local_id(0)] += logYhere;\n"
      // "    if(zeroCol) y[DstartObs + zeroCol + Npad * Dobs] = logYhere;\n"
      "  }// for Dobs\n"
      "  if(get_local_id(0)==0){\n"
      "    for(Dobs=1;Dobs < get_local_size(0);Dobs++){\n"
      "     sumLogY[0] += sumLogY[Dobs];\n"
      "    }// for Dobs\n"
      "    if(get_global_id(0)==0){\n"
      "     for(Dboxcox=0;Dboxcox < Nboxcox;Dboxcox++){\n"
      // " //   jacobian = -2*(BoxCox-1)* sum(log(yXcpu[,1]))\n"
      "      jacobian[DstartBoxcox + Dboxcox] =  -2*(param[DstartBoxcox + Dboxcox]-1)*sumLogY[0];\n"
      "     }// for Dboxcox\n"
      "    }// if global0\n"
      "  }// if local0\n"
      "}// if global and group\n";
    
    result +=
      "for(Dboxcox = get_global_id(1)+1; Dboxcox < Nboxcox; Dboxcox+= get_global_size(1)){\n"   // +1
      " boxcoxHere = param[DstartBoxcox + Dboxcox];\n"
      //" if(Dboxcox != zeroCol) {\n"
      "  for(Dobs=get_global_id(0);Dobs < Nobs; Dobs+= get_global_size(0)){\n"
      "//    transformed_y[ ,i] <- ((yXcpu[ ,1]^BoxCox[i]) - 1)/BoxCox[i]\n"
      "      y[DstartObs + Dboxcox + Npad * Dobs] = \n"
      "        (pow(y[DstartObs + Npad * Dobs],boxcoxHere)-1)/boxcoxHere;\n"
      "    }// for Dobs\n"
      //"  }// if zerocol\n"
      "}// for Dboxcox\n";
  }
  
  
  result +=   "}// kernel\n";
  return(result);
}


template<typename T> 
void addBoxcoxToData(
    viennacl::matrix_base<T> &yx,
    viennacl::vector_base<T> &boxcox,
    viennacl::vector_base<T> &jacobian,
    Rcpp::IntegerVector workgroupSize, 
    Rcpp::IntegerVector localSize, 
    Rcpp::IntegerVector NlocalCache, 
    const int ctx_id,
    Rcpp::IntegerVector verbose){
  
  viennacl::ocl::switch_context(ctx_id);
  viennacl::ocl::context ctx(viennacl::ocl::get_context(ctx_id));
  const int Ndatasets = boxcox.size();
  int boxcoxHasZero;
  if(  boxcox.size() > 1 ) {
    if(boxcox[1] == 0){
      //Rcpp::warning("second entry of boxcox parameters should be zero");
      boxcoxHasZero = 1;
    }else{
      boxcoxHasZero = 0;
    }
  }
  
  std::string theBoxcoxKernel = boxcoxKernelString<T>(
    NlocalCache[0],
               1, yx.size1(), Ndatasets, 
               yx.internal_size2(),
               boxcoxHasZero);
  
  if(verbose[0]>=3) {
    Rcpp::Rcout << "\n" << theBoxcoxKernel << "\n";
  }
  viennacl::ocl::program & my_prog_boxcox = viennacl::ocl::current_context().add_program(theBoxcoxKernel, "mkb");
  
  viennacl::ocl::kernel & boxcoxKernel = my_prog_boxcox.get_kernel("boxcox");
  boxcoxKernel.global_work_size(0, (cl_uint) (workgroupSize[0] ) );
  boxcoxKernel.global_work_size(1, (cl_uint) (workgroupSize[1] ) );
  boxcoxKernel.local_work_size(0, (cl_uint) (localSize[0]));
  boxcoxKernel.local_work_size(1, (cl_uint) (localSize[1]));
  
  viennacl::ocl::enqueue(
    boxcoxKernel(
      yx, boxcox, jacobian));
  clFinish(boxcoxKernel.context().get_queue().handle().get());
  
}











template<typename T> 
void likfitGpuP(viennacl::matrix_base<T> &yx, 
                viennacl::matrix_base<T> &coords, 
                viennacl::matrix_base<T> &params,
                //viennacl::matrix_base<T> &betas,
                viennacl::matrix_base<T> &ssqY,//viennacl::matrix_base<T> &aTDinvb_beta,
                viennacl::matrix_base<T> &XVYXVX,
                viennacl::matrix_base<T> &ssqBetahat,//viennacl::matrix_base<T> &ssqBeta,
                viennacl::vector_base<T> &detVar,
                viennacl::vector_base<T> &detReml,
                int Ndatasets,
                Rcpp::IntegerVector NparamPerIter,
                Rcpp::IntegerVector workgroupSize, 
                Rcpp::IntegerVector localSize, 
                Rcpp::IntegerVector NlocalCache, 
                const int ctx_id, 
                Rcpp::IntegerVector verbose,
                viennacl::matrix_base<T> &ssqYX,
                viennacl::matrix_base<T> &ssqYXcopy,
                viennacl::matrix_base<T> &LinvYX,
                viennacl::matrix_base<T> &QinvSsqYx,
                viennacl::matrix_base<T> &cholXVXdiag, 
                viennacl::matrix_base<T> &Vbatch,
                viennacl::matrix_base<T> &cholDiagMat){       
  //viennacl::matrix_base<T> &b_beta
  //viennacl::matrix_base<T> &aTDinvb_beta,
  //viennacl::matrix_base<T> &aTDinvb_beta_diag
  

  int Nobs = yx.size1();
  int Nparams = params.size1();
  int Ncovariates = yx.size2() - Ndatasets;
  int Ncol = yx.size2();
  
  
  int Niter = ceil( ( (T) Nparams) / ((T) NparamPerIter[0]));
  int Diter, Dy1, Dy2;
  int endThisIteration;
  int DiterIndex, NthisIteration;
  //int verboseMatern = verbose[0]>1;
  

  
  viennacl::ocl::switch_context(ctx_id);
  viennacl::ocl::context ctx(viennacl::ocl::get_context(ctx_id));
  
  
  int NgroupsOfParameters = workgroupSize[1]/localSize[1];
  int NparametersPerGroup = std::max(localSize[1], 
                                     ( (int) ceil( ((T) NparamPerIter[0]) / ((T) NgroupsOfParameters) ) )
  );
  
  if(NlocalCache[0] < (NlocalParams * NparametersPerGroup)){
    Rcpp::warning("NlocalCache too small for this number of parameters");
  }
  
  viennacl::ocl::local_mem localMemory(NlocalCache[0] *sizeof(yx(0,0) ) );
  

  //  viennacl::matrix<T> Vbatch(NparamPerIter[0]*Nobs, Nobs);
  //  viennacl::matrix<T> cholDiagMat(NparamPerIter[0], Nobs);
  //    viennacl::matrix<T> LinvYX(NparamPerIter[0]*Nobs, yx.size2());
  //    viennacl::matrix<T> ssqYX(NparamPerIter[0]*yx.size2(), yx.size2());
  
  //  viennacl::matrix<T> cholXVXdiag(NparamPerIter[0], Ncovariates);
  //  viennacl::matrix<T> QinvSsqYx(NparamPerIter[0]*Ncovariates, Ndatasets);
  
  fill22params(params);
  
  
  /* 
   * compute boxcox and jacobian
   */
  
  // create and compile kernels
  int Ncell = Nobs * (Nobs - 1)/2, maxIter = 1500;
  
  std::string maternClString = maternBatchKernelString<T>(
    maxIter,
    Nobs, 
    Ncell, 
    NparamPerIter[0],//NmatrixMax
                 Vbatch.internal_size2(), //NpadVbatch
                 Vbatch.internal_size2()*Nobs, //NpadBetweenMatrices,
                 coords.internal_size2(), //NpadCoords
                 params.internal_size2(), //NpadParams,
                 localSize[0], //Nlocal0 
                          //    NlocalParams * localSize[1] * (1+ NparamPerIter[0] * localSize[1] / workgroupSize[1]), // int NlocalParamsCache
                          1L, 1L, 1L, 0L); // assignUpper, assignLower, assignDiagonals, assignDistUpper
  
  int allowOverflow = ( ((int) Vbatch.size2() ) > NlocalCache[0] );

    
  
  std::string cholClString = cholBatchKernelString<T>(
    0, // colstart
    Nobs, // colend
    Nobs, // N
    Vbatch.internal_size2(), // Npad
    cholDiagMat.internal_size2(), // NpadDiag
    Vbatch.size2() * Vbatch.internal_size2(),// NpadBetweenMatrices,
    0,//NstartA
    0,// NstartD
    NlocalCache, //Ncache
    localSize, //Nlocal
    1,//allowOverflow, // allowoverflow
    1 // do log determinant
  );
  
  std::string cholXVXkernelString = cholBatchKernelString<T>(
    0, // colstart
    Ncovariates, // colend
    Ncovariates, // N
    ssqYX.internal_size2(), // Npad
    cholXVXdiag.internal_size2(), // NpadDiag
    ssqYX.internal_size2() * yx.size2(),// NpadBetweenMatrices, 
    Ndatasets * ssqYX.internal_size2() + Ndatasets,//NstartA start at entry ssqYX[Ndatasets, Ndatasets]
    0,// NstartD
    NlocalCache, //Ncache
    localSize, //Nlocal
    1, // allowoverflow
    1 // do log determinant
  );
  
  
  
  int Ngroups1 = static_cast<T>(workgroupSize[1]) / static_cast<T>(localSize[1]);
  int NcolsPerGroup = std::ceil( static_cast<T>(Ncol) / static_cast<T>(Ngroups1));
  int NlocalCacheSum = localSize[0] * localSize[1] * NcolsPerGroup;
  int NrowsToCache = std::floor(static_cast<T>(NlocalCache[0] - NlocalCacheSum) / static_cast<T>(NcolsPerGroup));
  int NlocalCacheC = NcolsPerGroup * NrowsToCache;
  
  if(NrowsToCache < 0 || (NlocalCache[0] - localSize[0]*localSize[1]) < 0) {
    Rcpp::warning("NlocalCache too small for this number of work groups");
  }
  std::string backsolveString = backsolveBatchString<T>(
    1,// sameB,
    1,// diagIsOne,
    Nobs,// Nrow, 
    Ncol,
    LinvYX.internal_size2(),// NpadC, 
    Vbatch.internal_size2(),// NpadA, 
    yx.internal_size2(),// NpadB,
    LinvYX.internal_size2()*Nobs,// NpadBetweenMatricesC,
    Vbatch.internal_size2()*Nobs,// NpadBetweenMatricesA,
    yx.internal_size2()*Nobs,// NpadBetweenMatricesB,
    0,// NstartC,
    0,// NstartA,
    0,// NstartB,
    NrowsToCache, 
    NcolsPerGroup,
    NlocalCacheC,  
    NlocalCacheSum,   
    localSize[0] * localSize[1]//NpadBetweenMatricesSum 
  );  
  
  // QinvSsqYx = Qinv SsqYx[-(1:Ndatasets), 1:Ncovariates]
  // Qinv stored in ssqYX[-(1:Ndatasets), -(1:Ndatasets)]
  Ncol = Ndatasets;
  NcolsPerGroup = std::ceil( static_cast<T>(Ncol) / static_cast<T>(Ngroups1));
  NrowsToCache = std::floor(static_cast<T>(NlocalCache[0]) /static_cast<T>(NcolsPerGroup));
  
  
  // backsolve QinvSsqYx = Q^(-1) ssqYX[(Ndatasets+1):nrow(ssqYX),1:Ndatasets]  
  // Ncovariates by Ndatasets
  std::string backsolveSsqYxString = backsolveBatchString<T>(
    0,// sameB,
    1,// diagIsOne,
    Ncovariates,// Nrow, 
    Ndatasets, // Ncol
    QinvSsqYx.internal_size2(),// NpadC, 
    ssqYX.internal_size2(),// NpadA, 
    ssqYX.internal_size2(),// NpadB,
    Ncovariates* QinvSsqYx.internal_size2(),// NpadBetweenMatricesC,
    ssqYX.size2()*ssqYX.internal_size2(),// NpadBetweenMatricesA,
    ssqYX.size2()*ssqYX.internal_size2(),// NpadBetweenMatricesB,
    0,// NstartC,
    Ndatasets * ssqYX.internal_size2() + Ndatasets,// NstartA,
    Ndatasets * ssqYX.internal_size2(),// NstartB,
    NrowsToCache, 
    NcolsPerGroup,
    NcolsPerGroup * NrowsToCache,// NlocalCacheC,  
    localSize[0] * localSize[1] * NcolsPerGroup,// NlocalCacheSum,   
    localSize[0] * localSize[1]//NpadBetweenMatricesSum 
  );
  
  
  const int NlocalCacheD = std::max( 0, (NlocalCache[0] - localSize[0]*localSize[1])/2 );
  
  std::string crossprodKernelString = crossprodBatchString<T>(
    Nobs,//const int Nrow, 
    yx.size2(),//const int Ncol,
    //    const int Nmatrix, 
    ssqYX.internal_size2(),//const int NpadC,
    LinvYX.internal_size2(),//const int NpadA,
    cholDiagMat.internal_size2(),//const int NpadD, // set to zero to omit D
    1, //const int invertD, // set to 1 for A^T D^(-1) A
    0, // not only the diagonals
    0,//const int NstartC,  
    0,//const int NstartA,  
    0,//const int NstartD,  
    ssqYX.internal_size2()*yx.size2(), //const int NpadBetweenMatricesC,
    LinvYX.internal_size2()*Nobs, //const int NpadBetweenMatricesA,
    NlocalCacheD // NlocalCacheA, localSize// Nlocal// cache a Nlocal[0] by Nlocal[1] submatrix of C
  );
  
  // ssqBetahat     p*Ndatasets
  std::string crossprodSsqYxKernelString = crossprodBatchString<T>(
    Ncovariates,//const int Nrow, 
    Ndatasets,//const int Ncol,
    //    const int Nmatrix, 
    ssqBetahat.internal_size2(),//const int NpadC, 
    QinvSsqYx.internal_size2(),//const int NpadA,
    cholXVXdiag.internal_size2(),//const int NpadD, // set to zero to omit D
    1, //const int invertD, // set to 1 for A^T D^(-1) A
    1, // only diagonals
    0,//const int NstartC,  
    0,//const int NstartA,  
    0,//const int NstartD,  
    ssqBetahat.internal_size2(), //const int NpadBetweenMatricesC
    QinvSsqYx.internal_size2()*Ncovariates, //const int NpadBetweenMatricesA,  made changes here
    NlocalCacheD // NlocalCacheA, localSize// Nlocal// cache a Nlocal[0] by Nlocal[1] submatrix of C
  );
  
  
  std::string extractSomeDiagKernelString = extract_some_diag_string<T>(//NparamPerIter[0],  
    Ndatasets,
    ssqY.internal_size2(), 
    ssqYX.internal_size2(), 
    ssqYX.internal_size2()*ssqYX.size2()
  );
  
  
  
  
  
std::string extractBlockKernelString = extract_block_string<T>(
    Ncovariates,    // Ncovariates
    Ndatasets,    // Ndatasets
    XVYXVX.internal_size2(),  // NpadColXVYXVX
    ssqYX.internal_size2(),  // NpadColYX
    XVYXVX.internal_size2()*Ncovariates,  // NpadBetweenMatricesXVYXVX = NpadColXVYXVX * Ncovariates
    ssqYX.internal_size2()*ssqYX.size2()  // NpadBetweenMatricesYX
  );  
  
  if(verbose[0]>1){
    Rcpp::Rcout << extractBlockKernelString << "\n";
  }
  
  viennacl::ocl::program & my_prog_matern = viennacl::ocl::current_context().add_program(maternClString, "mykernelmatern");
  viennacl::ocl::kernel & maternKernel = my_prog_matern.get_kernel("maternBatch");
  // dimension 0 is cell, dimension 1 is matrix
  if(verbose[0]){
  Rcpp::Rcout << "workgroupSize\n" << workgroupSize << "\n";
  Rcpp::Rcout << "localSize\n" << localSize << "\n";  
  }
  
  maternKernel.global_work_size(0, workgroupSize[0] ); 
  maternKernel.global_work_size(1, workgroupSize[1] ); 
  maternKernel.local_work_size(0, localSize[0]);
  maternKernel.local_work_size(1, localSize[1]);
  
  
  viennacl::ocl::program & my_prog_chol = viennacl::ocl::current_context().add_program(cholClString, "mykernelchol");
  viennacl::ocl::kernel & cholKernel = my_prog_chol.get_kernel("cholBatch");
  

  cholKernel.global_work_size(0, workgroupSize[0] ); 
  cholKernel.global_work_size(1, localSize[1] ); 
  cholKernel.local_work_size(0,  localSize[0]);
  cholKernel.local_work_size(1,  localSize[1]);
  
  viennacl::ocl::program & my_prog_crossprodSsqYx = viennacl::ocl::current_context().add_program(crossprodSsqYxKernelString, "mykernelcrossprodssqyx");
  viennacl::ocl::program & my_prog_extractBlock = viennacl::ocl::current_context().add_program(extractBlockKernelString, "mykernelextract_Block");
  viennacl::ocl::program & my_prog_backsolve = viennacl::ocl::current_context().add_program(backsolveString, "mykernelbacksolve");
  viennacl::ocl::program & my_prog_crossprod = viennacl::ocl::current_context().add_program(crossprodKernelString, "mykernelcrossprod");
  viennacl::ocl::program & my_prog_cholxvx = viennacl::ocl::current_context().add_program(cholXVXkernelString, "mykernelcholxvx");
  viennacl::ocl::program & my_prog_backsolveSsqYx = viennacl::ocl::current_context().add_program(backsolveSsqYxString, "mybacksolvessqyx");
  viennacl::ocl::program & my_prog_extractSomeDiag = viennacl::ocl::current_context().add_program(extractSomeDiagKernelString, "mykernelextract_some_diag");
  
  viennacl::ocl::kernel & cholXvxKernel = my_prog_cholxvx.get_kernel("cholBatch");
  viennacl::ocl::kernel & backsolveKernel = my_prog_backsolve.get_kernel("backsolveBatch");
  viennacl::ocl::kernel & crossprodKernel = my_prog_crossprod.get_kernel("crossprodBatch");
  viennacl::ocl::kernel & backsolveSsqYxKernel = my_prog_backsolveSsqYx.get_kernel("backsolveBatch");
  viennacl::ocl::kernel & crossprodSsqYxKernel = my_prog_crossprodSsqYx.get_kernel("crossprodBatch");
  viennacl::ocl::kernel & extractSomeDiagKernel = my_prog_extractSomeDiag.get_kernel("extract_some_diag");
  viennacl::ocl::kernel & extractBlockKernel = my_prog_extractBlock.get_kernel("extract_block");
  

  cholXvxKernel.global_work_size(0, workgroupSize[0] ); 
  cholXvxKernel.global_work_size(1, localSize[1] ); 
  cholXvxKernel.local_work_size(0,  localSize[0]);
  cholXvxKernel.local_work_size(1,  localSize[1]);
  
  backsolveKernel.global_work_size(0, workgroupSize[0] ); 
  backsolveKernel.global_work_size(1, workgroupSize[1] ); 
  backsolveKernel.local_work_size(0, localSize[0]);
  backsolveKernel.local_work_size(1, localSize[1]);
  
  crossprodKernel.global_work_size(0, workgroupSize[0] ); 
  crossprodKernel.global_work_size(1, workgroupSize[1] ); 
  crossprodKernel.local_work_size(0, localSize[0]);
  crossprodKernel.local_work_size(1, localSize[1]);
  
  backsolveSsqYxKernel.global_work_size(0, workgroupSize[0] ); 
  backsolveSsqYxKernel.global_work_size(1, workgroupSize[1] ); 
  backsolveSsqYxKernel.local_work_size(0, localSize[0]);
  backsolveSsqYxKernel.local_work_size(1, localSize[1]);
  
  crossprodSsqYxKernel.global_work_size(0, workgroupSize[0] ); 
  crossprodSsqYxKernel.global_work_size(1, workgroupSize[1] ); 
  crossprodSsqYxKernel.local_work_size(0, localSize[0]);
  crossprodSsqYxKernel.local_work_size(1, localSize[1]);
  
  extractSomeDiagKernel.global_work_size(0, workgroupSize[0] ); 
  extractSomeDiagKernel.global_work_size(1, workgroupSize[1] );   
  extractSomeDiagKernel.local_work_size(0, 1L);
  extractSomeDiagKernel.local_work_size(1, 1L);
  

  extractBlockKernel.global_work_size(0, 1L);//workgroupSize[0] ); 
  extractBlockKernel.global_work_size(1, 1L);//workgroupSize[1] );   
  extractBlockKernel.local_work_size(0, 1L);
  extractBlockKernel.local_work_size(1, 1L);
  
  if(verbose[0]) {
    Rcpp::Rcout << "\n" << " Nparams " << 
      Nparams << " NparamsPerIter " <<  NparamPerIter[0] <<
        " Niter " << Niter << " Ncovariates " << Ncovariates << " Ndatasets " << Ndatasets << " NlocalCacheD " << NlocalCacheD << "\n";
  }
  
  ///////////////////////////Loop starts !!!//////////////////////////////////////////////////////////////////////////
  
  viennacl::ocl::command_queue theQueue = maternKernel.context().get_queue();
  /*
   if(verbose[1]==1){  // given betas
   for (Diter=0,DiterIndex=0; 
   Diter< Niter; 
   Diter++,DiterIndex += NparamPerIter[0]){
   
   
   endThisIteration = std::min(DiterIndex + NparamPerIter[0], Nparams);
   NthisIteration = endThisIteration - DiterIndex;
   
   if(verbose[0]>1) {
   Rcpp::Rcout << "\n" << " Diter " << 
   Diter << " endThisIteration " <<  endThisIteration <<
   " NthisIteration " << NthisIteration << " localMemorySize " << localMemory.size() << " DiterIndex " << DiterIndex << "\n";
   }
   // matern
   viennacl::ocl::enqueue(maternKernel(Vbatch, coords, params, 
   localMemory,
   DiterIndex, NthisIteration),
   theQueue);
   
   // cholesky Vbatch=LDL^T, cholesky decomposition
   viennacl::ocl::enqueue(cholKernel(Vbatch, cholDiagMat, 
   localMemory, 
   NthisIteration, detVar, DiterIndex),
   theQueue);
   
   // backsolve  LinvYX = L^(-1) YX,   Nobs by (Ndatasets + Ncovariates)
   viennacl::ocl::enqueue(backsolveKernel(LinvYX, Vbatch, yx, 
   localMemory,
   NthisIteration),
   theQueue);
   
   // crossprod  ssqYX = YX^Y L^(-1)T D^(-1) L^(-1) YX  square matrix, (Ndatasets + Ncovariates)
   viennacl::ocl::enqueue(crossprodKernel(ssqYX, LinvYX, cholDiagMat, 0, NthisIteration),
   theQueue);
   
   
   // save diagonals of ssqYX to ssqY
   viennacl::ocl::enqueue(extractSomeDiagKernel(ssqY, ssqYX, DiterIndex, NthisIteration),
   theQueue);  
   
   // for(Dy1 = 0; Dy1 < Ndatasets; Dy1++) {
   //   for(Dy2 = 0; Dy2 < NthisIteration; Dy2++) {
   //     ssqY(DiterIndex + Dy2,Dy1) = ssqYX( Dy2 * ssqYX.size2() + Dy1, Dy1);
   //   }
   // }
   
   // for(Dy1 = 0; Dy1 < ssqYXcopy.size1(); Dy1++) {
   //   for(Dy2 = 0; Dy2 < ssqYXcopy.size2(); Dy2++) {
   //     ssqYXcopy(Dy1, Dy2) = ssqYX(Dy1, Dy2);
   //   }}
   
   
   // save bottom block of ssqYX to XVYXVX
   viennacl::ocl::enqueue(extractBlockKernel(XVYXVX, ssqYX, DiterIndex, NthisIteration),
   theQueue);  
   
   
   
   // cholesky X^T V^(-1) X = QPQ^T, save determinant as detReml, changes Ncovariates by Ncovariates part
   viennacl::ocl::enqueue(cholXvxKernel(ssqYX, cholXVXdiag, localMemory, NthisIteration, detReml, DiterIndex),
   theQueue);
   
   // if(verbose[0]>3) {
   //   Rcpp::Rcout << "cxy";
   // }
   
   // backsolve QinvSsqYx = Q^(-1) ssqYX[(Ndatasets+1):nrow(ssqYX),1:Ndatasets]  
   // Ncovariates by Ndatasets
   viennacl::ocl::enqueue(backsolveSsqYxKernel(QinvSsqYx, ssqYX, ssqYX, NthisIteration),
   theQueue);
   
   
   
   // crossprod QinvSsqYx^T P^(-1) QinvSsqYx,   Ndatasets by Ndatasets
   viennacl::ocl::enqueue(crossprodSsqYxKernel(ssqBetahat, QinvSsqYx, cholXVXdiag, DiterIndex, NthisIteration),
   theQueue); 
   
   
   
   // gemmBatch2  a^TD^(-1)b * beta = aTDinvb_beta         
   viennacl::ocl::enqueue(aTDinvb_betaKernel(ssqYX, betas, aTDinvb_beta, DiterIndex, NthisIteration), //nRowBatch
   theQueue);     
   
   // gemmBatch2   b*beta    
   viennacl::ocl::enqueue(b_betasKernel(LinvYX, betas, b_beta, 0, NthisIteration), //nRowBatch
   theQueue);   
   
   // ssqBeta_crossprod      ssqBeta = (X beta)^T V^(-1) (X beta) = (b_beta)^T D^(-1) (b_beta) 
   viennacl::ocl::enqueue(ssqBetaKernel(ssqBeta, b_beta, cholDiagMat, DiterIndex, NthisIteration), //nRowBatch
   theQueue);
   
   // if(Diter ==1)
   // Rcpp::Rcout << "crossprod_ssqBeta_KernelString\n" << crossprod_ssqBeta_KernelString << "\n";
   
   
   if(verbose[0]>1) {
   Rcpp::Rcout << "\n" << "Diter " << Diter <<" DiterIndex " << DiterIndex << " endThisIteration " << 
   endThisIteration << " Nthisiteration " << NthisIteration  << "\n";
   }
   } // Diter
   }else{
   */

  for (Diter=0,DiterIndex=0; 
       Diter< Niter; 
       Diter++,DiterIndex += NparamPerIter[0]){
    
    
    endThisIteration = std::min(DiterIndex + NparamPerIter[0], Nparams);
    NthisIteration = endThisIteration - DiterIndex;
    

    if(verbose[0]>=2) {
      Rcpp::Rcout << "\n" << " Diter " << 
        Diter << " endThisIteration " <<  endThisIteration <<
          " NthisIteration " << NthisIteration << " localMemorySize " << localMemory.size() << " DiterIndex " << DiterIndex <<"\n";
    }
    // matern
    viennacl::ocl::enqueue(maternKernel(Vbatch, coords, params, 
                                        localMemory,
                                        DiterIndex, NthisIteration),
                                        theQueue);

    if( verbose[1]>=2){
      Rcpp::Rcout << "maternkernel finished\n" << "\n\n";
    }
    
    // Vbatch=LDL^T, cholesky decomposition
    viennacl::ocl::enqueue(cholKernel(Vbatch, cholDiagMat, 
                                      localMemory, NthisIteration, 
                                      detVar, DiterIndex),
                                      theQueue);

    
    if( verbose[1]>=2){
      Rcpp::Rcout << "cholKernel finished\n" << "\n\n";
    }
    
    
    // backsolve  LinvYX = L^(-1) YX,   Nobs by (Ndatasets + Ncovariates)
    viennacl::ocl::enqueue(backsolveKernel(LinvYX, Vbatch, yx,
                                           localMemory,
                                           NthisIteration),
                                           theQueue);
    
    if( verbose[1]>=2){
      Rcpp::Rcout << "backsolveKernel finished\n" << "\n\n";
    }     
    
    
    // crossprod ssqYX = YX^Y L^(-1)T D^(-1) L^(-1) YX  square matrix, (Ndatasets + Ncovariates)
    viennacl::ocl::enqueue(crossprodKernel(ssqYX, LinvYX, cholDiagMat,
                                           localMemory,
                                           0, NthisIteration),
                                           theQueue);
    
    if( verbose[1]>=2){
      Rcpp::Rcout << "crossprodKernel finished\n" << "\n\n";
    }
    

    /*    
    for(Dy1 = 0; Dy1 < ssqYXcopy.size1(); Dy1++) {
      for(Dy2 = 0; Dy2 < ssqYXcopy.size2(); Dy2++) {
        ssqYXcopy(Dy1, Dy2) = ssqYX(Dy1, Dy2);
      }
    }
     */
  // the above was because the copying below didnt always work, seems ok now    
    ssqYXcopy = ssqYX;
    

    // save diagonals of ssqYX to ssqY
    viennacl::ocl::enqueue(extractSomeDiagKernel(ssqY, ssqYX, DiterIndex, NthisIteration),  // put 0 on diterindex
                           theQueue);
    if( verbose[1]>=2){
      Rcpp::Rcout << "extractSomeDiagKernel finished\n" << "\n\n";
    }     
    

    // save bottom block of ssqYX to XVYXVX
    viennacl::ocl::enqueue(extractBlockKernel(XVYXVX, ssqYX, DiterIndex, NthisIteration),
                           theQueue);
    if( verbose[1]>=2){
      Rcpp::Rcout << "extractBlockKernel finished\n" << "\n\n";
    }     
    
    if( verbose[0]>=3){
      Rcpp::Rcout << "maternClString\n" << maternClString << "\n";
      Rcpp::Rcout << "cholClString\n" << cholClString << "\n";
    }
    

    // cholesky X^T V^(-1) X = QPQ^T, save determinant as detReml, changes Ncovariates by Ncovariates part
    viennacl::ocl::enqueue(cholXvxKernel(ssqYX, cholXVXdiag, localMemory,
                                         NthisIteration, detReml, DiterIndex),   // put 0
                                         theQueue);
    if(Diter ==1 & verbose[1]>=2){
      Rcpp::Rcout << "cholesky ssqX finished\n" << "\n\n";
    }  
    
    // backsolve QinvSsqYx = Q^(-1) ssqYX[(Ndatasets+1):nrow(ssqYX),1:Ndatasets]  , Ncovariates by Ndatasets
    viennacl::ocl::enqueue(backsolveSsqYxKernel(QinvSsqYx, ssqYX, ssqYX, localMemory, NthisIteration),
                           theQueue);

    if(Diter ==1 & verbose[1]>=2){
      Rcpp::Rcout << "backsolve QinvSsqYx for temp3 finished\n" << "\n\n";
    }  
    
    
    // crossprod QinvSsqYx^T P^(-1) QinvSsqYx,   NthisIteration by Ndatasets    ssqBetahat
    viennacl::ocl::enqueue(
      crossprodSsqYxKernel(ssqBetahat, QinvSsqYx, cholXVXdiag,
                           localMemory,
                           DiterIndex, NthisIteration),
                           theQueue);
    if( verbose[1]>=2){
      Rcpp::Rcout << "crossprod for ssqBetahat finished\n" << "\n\n";
    }  

        // do loglik calculation
    // copy to cpu
    
    
    
    
  }
  
  

  clFinish(theQueue.handle().get());
  
  
}


















template<typename T> 
void likfitGpuP_Templated(
    Rcpp::S4 yx,
    Rcpp::S4 coords,
    Rcpp::S4 params,
    Rcpp::S4 boxcox,
    //Rcpp::S4 betas,
    Rcpp::S4 ssqY, //Rcpp::S4 aTDinvb_beta,
    Rcpp::S4 XVYXVX,
    Rcpp::S4 ssqBetahat, //Rcpp::S4 ssqBeta,
    Rcpp::S4 detVar,
    Rcpp::S4 detReml,
    Rcpp::S4 jacobian,
    Rcpp::IntegerVector NparamPerIter,
    Rcpp::IntegerVector workgroupSize,
    Rcpp::IntegerVector localSize,
    Rcpp::IntegerVector NlocalCache,
    Rcpp::IntegerVector verbose,
    Rcpp::S4 ssqYX,        
    Rcpp::S4 ssqYXcopy,
    Rcpp::S4 LinvYX, 
    Rcpp::S4 QinvSsqYx, 
    Rcpp::S4 cholXVXdiag,
    Rcpp::S4 varMat,
    Rcpp::S4 cholDiagMat
){  //Rcpp::S4 b_beta, Rcpp::S4 aTDinvb_beta_diag
  
  
  const bool BisVCL=1;
  const int ctx_id = INTEGER(yx.slot(".context_index"))[0]-1;
  std::shared_ptr<viennacl::matrix<T> > yxGpu = getVCLptr<T>(yx.slot("address"), BisVCL, ctx_id);
  std::shared_ptr<viennacl::matrix<T> > coordsGpu = getVCLptr<T>(coords.slot("address"), BisVCL, ctx_id);
  std::shared_ptr<viennacl::matrix<T> > paramsGpu = getVCLptr<T>(params.slot("address"), BisVCL, ctx_id);
  //std::shared_ptr<viennacl::matrix<T> > betasGpu = getVCLptr<T>(betas.slot("address"), BisVCL, ctx_id);
  std::shared_ptr<viennacl::matrix<T> > ssqYGpu = getVCLptr<T>(ssqY.slot("address"), BisVCL, ctx_id);
  //std::shared_ptr<viennacl::matrix<T> > aTDinvb_betaGpu = getVCLptr<T>(aTDinvb_beta.slot("address"), BisVCL, ctx_id);
  std::shared_ptr<viennacl::matrix<T> > XVYXVXGpu = getVCLptr<T>(XVYXVX.slot("address"), BisVCL, ctx_id);
  std::shared_ptr<viennacl::matrix<T> > ssqBetahatGpu = getVCLptr<T>(ssqBetahat.slot("address"), BisVCL, ctx_id);
  //std::shared_ptr<viennacl::matrix<T> > ssqBetaGpu = getVCLptr<T>(ssqBeta.slot("address"), BisVCL, ctx_id);
  std::shared_ptr<viennacl::vector_base<T> > detVarGpu = getVCLVecptr<T>(detVar.slot("address"), BisVCL, ctx_id);
  std::shared_ptr<viennacl::vector_base<T> > detRemlGpu = getVCLVecptr<T>(detReml.slot("address"), BisVCL, ctx_id);
  std::shared_ptr<viennacl::vector_base<T> > boxcoxGpu = getVCLVecptr<T>(boxcox.slot("address"), BisVCL, ctx_id);
  std::shared_ptr<viennacl::vector_base<T> > jacobianGpu = getVCLVecptr<T>(jacobian.slot("address"), BisVCL, ctx_id);
  
  std::shared_ptr<viennacl::matrix<T> > ssqYXgpu = getVCLptr<T>(ssqYX.slot("address"), BisVCL, ctx_id);
  std::shared_ptr<viennacl::matrix<T> > ssqYXcopyGpu= getVCLptr<T>(ssqYXcopy.slot("address"), BisVCL, ctx_id);
  std::shared_ptr<viennacl::matrix<T> > LinvYXgpu = getVCLptr<T>(LinvYX.slot("address"), BisVCL, ctx_id);
  std::shared_ptr<viennacl::matrix<T> > QinvSsqYxgpu = getVCLptr<T>(QinvSsqYx.slot("address"), BisVCL, ctx_id);
  std::shared_ptr<viennacl::matrix<T> > cholXVXdiaggpu = getVCLptr<T>(cholXVXdiag.slot("address"), BisVCL, ctx_id);
  std::shared_ptr<viennacl::matrix<T> > varMatgpu = getVCLptr<T>(varMat.slot("address"), BisVCL, ctx_id);
  std::shared_ptr<viennacl::matrix<T> > cholDiagMatgpu= getVCLptr<T>(cholDiagMat.slot("address"), BisVCL, ctx_id);
  //std::shared_ptr<viennacl::matrix<T> > b_betagpu= getVCLptr<T>(b_beta.slot("address"), BisVCL, ctx_id);
  //  std::shared_ptr<viennacl::matrix<T> > aTDinvb_betagpu = getVCLptr<T>(aTDinvb_beta.slot("address"), BisVCL, ctx_id);
  //  std::shared_ptr<viennacl::matrix<T> > aTDinvb_beta_diaggpu = getVCLptr<T>(aTDinvb_beta_diag.slot("address"), BisVCL, ctx_id);

  
  addBoxcoxToData<T>(
    *yxGpu,
    *boxcoxGpu,
    *jacobianGpu,
    workgroupSize, 
    localSize,
    NlocalCache,
    ctx_id,
    verbose);
  
  

  
  likfitGpuP<T>(
    *yxGpu, 
    *coordsGpu, 
    *paramsGpu, 
    //*betasGpu,
    *ssqYGpu, //*aTDinvb_betaGpu,
    *XVYXVXGpu,
    *ssqBetahatGpu,//*ssqBetaGpu,
    *detVarGpu, 
    *detRemlGpu,
    (*boxcoxGpu).size(),// Ndatasets
    NparamPerIter,
    workgroupSize, 
    localSize, 
    NlocalCache, 
    ctx_id, 
    verbose, 
    *ssqYXgpu, 
    *ssqYXcopyGpu, 
    *LinvYXgpu, 
    *QinvSsqYxgpu,
    *cholXVXdiaggpu, 
    *varMatgpu, 
    *cholDiagMatgpu);
    // *b_betagpu
    // *aTDinvb_betagpu,
    // *aTDinvb_beta_diaggpu
}




















//[[Rcpp::export]]
void likfitGpu_BackendP(
    Rcpp::S4 yx,   //1
    Rcpp::S4 coords,  //2
    Rcpp::S4 params,  //3
    Rcpp::S4 boxcox,  //4//Rcpp::S4 betas,   //5    not used
    Rcpp::S4 ssqY,  //6   //Rcpp::S4 aTDinvb_beta, //7
    Rcpp::S4 XVYXVX,  //6 
    Rcpp::S4 ssqBetahat, //9//Rcpp::S4 ssqBeta,  //10
    Rcpp::S4 detVar, //11
    Rcpp::S4 detReml, //12
    Rcpp::S4 jacobian,  //10
    Rcpp::IntegerVector NparamPerIter,//14
    Rcpp::IntegerVector workgroupSize,//15
    Rcpp::IntegerVector localSize,//16
    Rcpp::IntegerVector NlocalCache,//17
    Rcpp::IntegerVector verbose,//15    verbose[2]=betasgiven, true or false
    Rcpp::S4 ssqYX, //16         col number must be exactly Ncovariates + Ndatasets  //
    Rcpp::S4 ssqYXcopy, //20   not really used? cannot exist an empty line here 
    Rcpp::S4 LinvYX, //21
    Rcpp::S4 QinvSsqYx, //22
    Rcpp::S4 cholXVXdiag,//23
    Rcpp::S4 varMat, //21     Vbatch
    Rcpp::S4 cholDiagMat //22
) { 
  // Rcpp::S4 b_beta
  //    Rcpp::S4 aTDinvb_beta,
  //    Rcpp::S4 aTDinvb_beta_diag
  
  
  Rcpp::traits::input_parameter< std::string >::type classVarR(RCPP_GET_CLASS(yx));
  std::string precision_type = (std::string) classVarR;
  
  
  if(precision_type == "fvclMatrix") {
    likfitGpuP_Templated<float>(  yx,
                                  coords,
                                  params,
                                  boxcox,//betas,
                                  ssqY,//aTDinvb_beta,
                                  XVYXVX,
                                  ssqBetahat,//ssqBeta,
                                  detVar,
                                  detReml,
                                  jacobian,
                                  NparamPerIter,
                                  workgroupSize,
                                  localSize,
                                  NlocalCache,
                                  verbose, ssqYX, //
                                  ssqYXcopy,
                                  LinvYX, QinvSsqYx, cholXVXdiag, varMat, cholDiagMat); //b_beta); 
  } 
  else if (precision_type == "dvclMatrix") {
    likfitGpuP_Templated<double>(  yx,
                                   coords,
                                   params,
                                   boxcox,//betas,
                                   ssqY,//aTDinvb_beta,
                                   XVYXVX,
                                   ssqBetahat,//ssqBeta,
                                   detVar,
                                   detReml,
                                   jacobian,
                                   NparamPerIter,
                                   workgroupSize,
                                   localSize,
                                   NlocalCache,
                                   verbose, ssqYX, //
                                   ssqYXcopy, 
                                   LinvYX, QinvSsqYx, cholXVXdiag, varMat, cholDiagMat); //b_beta); 
  } else {
    Rcpp::warning("class of var must be fvclMatrix or dvclMatrix");
  }
}

  
  





























