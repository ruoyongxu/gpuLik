#include "gpuRandom.hpp"


//#define DEBUGKERNEL
template <typename T> 
std::string matrix_plus_vectorString(const int Nrow, const int Ncol, 
                                     const int matrix_NpadCol, const int Result_NpadCol) {  //internal column size
  
  std::string typeString = openclTypeString<T>();  // type of the sum of log factorial
  
  std::string result = "";
  
  if(typeString == "double") {
    result += "\n#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
  }
  
  
  result +=
    "\n#define Nrow " + std::to_string(Nrow) + "\n"    
    "#define Ncol " + std::to_string(Ncol) + "\n"
    "#define matrix_NpadCol " + std::to_string(matrix_NpadCol) + "\n"
    "#define Result_NpadCol " + std::to_string(Result_NpadCol) + "\n";    
  
  
  result += 
    "\n\n__kernel void matrix_add_vector(\n"
    "  __global " + typeString + "* m,\n"  
    "  __global " + typeString + "* rv,\n"
    "  __global " + typeString + "* cv,\n"
    + typeString + " value,\n"
  "  __global " + typeString + "* result"  
  "){\n\n";  
  
  
  result += "int Drow, Dcol;\n";
  
  
  
  result += 
    "  for(Drow = get_global_id(0);   Drow < Nrow;  Drow+=get_global_size(0)){\n"
    "    for(Dcol = get_global_id(1);  Dcol < Ncol;   Dcol+=get_global_size(1)){\n"
    
    "  result[Drow*Result_NpadCol+Dcol] = m[Drow*matrix_NpadCol+Dcol] + rv[Drow] + cv[Dcol] + value;\n"
    
    "    } // end loop through columns\n"
    "  } // end loop through rows\n";
  
  
  
  result += 
    "}\n";
  
  
  return(result);
}










template<typename T> 
void matrix_vectors_sum(
    viennacl::matrix_base<T> &matrix,// viennacl::vector_base<int>  rowSum, viennacl::vector_base<int>  colSum,  
    viennacl::vector_base<T> &rowvector,
    viennacl::vector_base<T> &colvector,  
    T  constant,
    viennacl::matrix_base<T> &sum,
    Rcpp::IntegerVector numWorkItems,
    const int ctx_id) {
  
  if ((rowvector.size()!=matrix.size1()) && (colvector.size()!=matrix.size2())){
    Rcpp::Rcout << "Error: cannot do plus operation" << "\n\n";
    // return EXIT_FAILURE;
  }
  
  std::string KernelString = matrix_plus_vectorString<T>(
    sum.size1(), 
    sum.size2(),
    matrix.internal_size2(),
    sum.internal_size2()
  );
  
  viennacl::ocl::switch_context(ctx_id);
  viennacl::ocl::program & my_prog = viennacl::ocl::current_context().add_program(KernelString, "my_kernel");
  
#ifdef DEBUGKERNEL
  Rcpp::Rcout << KernelString << "\n\n";
#endif  
  
  viennacl::ocl::kernel &matrix_plus_vectorsKernel = my_prog.get_kernel("matrix_add_vector");
  matrix_plus_vectorsKernel.global_work_size(0, numWorkItems[0]);
  matrix_plus_vectorsKernel.global_work_size(1, numWorkItems[1]);
  matrix_plus_vectorsKernel.local_work_size(0, 1L);
  matrix_plus_vectorsKernel.local_work_size(1, 1L);
  
  
  viennacl::ocl::enqueue(matrix_plus_vectorsKernel(matrix, rowvector, colvector, constant, sum) );
  
  
  // return 1L;
  
}



template<typename T> 
void matrix_vector_sumTemplated(
    Rcpp::S4 matrixR,
    Rcpp::S4 rowvectorR,
    Rcpp::S4 colvectorR, 
    SEXP  constantR,
    Rcpp::S4 sumR,
    Rcpp::IntegerVector numWorkItems) {
  
  T constant = Rcpp::as<T>(constantR); 
  const bool BisVCL=1;
  const int ctx_id = INTEGER(sumR.slot(".context_index"))[0]-1;
  std::shared_ptr<viennacl::matrix<T> > matrix = getVCLptr<T>(matrixR.slot("address"), BisVCL, ctx_id);
  std::shared_ptr<viennacl::vector_base<T> > rowvector = getVCLVecptr<T>(rowvectorR.slot("address"), BisVCL, ctx_id);
  std::shared_ptr<viennacl::vector_base<T> > colvector = getVCLVecptr<T>(colvectorR.slot("address"), BisVCL, ctx_id);
  std::shared_ptr<viennacl::matrix<T> > sum = getVCLptr<T>(sumR.slot("address"), BisVCL, ctx_id);
  
  matrix_vectors_sum(*matrix, *rowvector, *colvector, constant, *sum, numWorkItems, ctx_id);
  
}





// [[Rcpp::export]]
void matrix_vector_sumBackend(
    Rcpp::S4 matrixR,
    Rcpp::S4 rowvectorR,
    Rcpp::S4 colvectorR,  
    SEXP  constantR,
    Rcpp::S4 sumR,
    Rcpp::IntegerVector numWorkItems) {
  
  Rcpp::traits::input_parameter< std::string >::type classVarR(RCPP_GET_CLASS(sumR));
  std::string precision_type = (std::string) classVarR;
  
  
  if(precision_type == "fvclMatrix") {
    return (matrix_vector_sumTemplated<float>(matrixR, rowvectorR, colvectorR, constantR, sumR,  numWorkItems));
  } else if (precision_type == "dvclMatrix") {
    return (matrix_vector_sumTemplated<double>(matrixR, rowvectorR, colvectorR, constantR, sumR,  numWorkItems));
  } else if (precision_type  == "ivclMatrix") {
    return( matrix_vector_sumTemplated<int>(matrixR, rowvectorR, colvectorR, constantR, sumR,  numWorkItems));
  }
  
}






















