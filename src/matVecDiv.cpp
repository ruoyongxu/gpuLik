#include "lgmlikFit.hpp"


//#define DEBUGKERNEL
template <typename T> 
std::string matrix_divide_vectorString(const int Nrow, 
                                       const int Ncol, 
                                       const int matrix_NpadCol, 
                                       const int Result_NpadCol) {  //internal column size
  
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
    "\n\n__kernel void matrix_divide_vector(\n"
    "  __global " + typeString + "* m,\n"  
    "  __global " + typeString + "* v,\n"
  "  __global " + typeString + "* result"  
  "){\n\n";  
  
  
  result += "int Drow, Dcol;\n";
  
  
  
  result += 
    "  for(Drow = get_global_id(0);   Drow < Nrow;  Drow+=get_global_size(0)){\n"
    "    for(Dcol = get_global_id(1);  Dcol < Ncol;   Dcol+=get_global_size(1)){\n"
    
    "  result[Drow*Result_NpadCol+Dcol] = m[Drow*matrix_NpadCol+Dcol]/ v[Drow];\n"
    
    "    } // end loop through columns\n"
    "  } // end loop through rows\n";
  
  
  
  result += 
    "}\n";
  
  
  return(result);
}










template<typename T> 
void matrix_vector_eledivide(
    viennacl::matrix_base<T> &matrix,// viennacl::vector_base<int>  rowSum, viennacl::vector_base<int>  colSum,  
    viennacl::vector_base<T> &rowvector,
    viennacl::matrix_base<T> &result,
    Rcpp::IntegerVector numWorkItems,
    const int ctx_id) {
  
  if (rowvector.size()!=matrix.size1()){
    Rcpp::Rcout << "Error: cannot do plus operation" << "\n\n";
    // return EXIT_FAILURE;
  }
  
  std::string KernelString = matrix_divide_vectorString<T>(
    matrix.size1(), 
    matrix.size2(),
    matrix.internal_size2(),
    result.internal_size2()
  );
  
  viennacl::ocl::switch_context(ctx_id);
  viennacl::ocl::program & my_prog = viennacl::ocl::current_context().add_program(KernelString, "my_kernel");
  
#ifdef DEBUGKERNEL
  Rcpp::Rcout << KernelString << "\n\n";
#endif  
  
  viennacl::ocl::kernel &matrix_divide_vectorKernel = my_prog.get_kernel("matrix_divide_vector");
  matrix_divide_vectorKernel.global_work_size(0, numWorkItems[0]);
  matrix_divide_vectorKernel.global_work_size(1, numWorkItems[1]);
  matrix_divide_vectorKernel.local_work_size(0, 1L);
  matrix_divide_vectorKernel.local_work_size(1, 1L);
  
  
  
  
  viennacl::ocl::enqueue(matrix_divide_vectorKernel(matrix, rowvector,result) );
  
  
  // return 1L;
  
}



template<typename T> 
void mat_vec_eledivideTemplated(
    Rcpp::S4 matrixR,
    Rcpp::S4 rowvectorR,
    Rcpp::S4 resultR,
    Rcpp::IntegerVector numWorkItems) {
  
  const bool BisVCL=1;
  const int ctx_id = INTEGER(matrixR.slot(".context_index"))[0]-1;
  std::shared_ptr<viennacl::matrix<T> > matrix = getVCLptr<T>(matrixR.slot("address"), BisVCL, ctx_id);
  std::shared_ptr<viennacl::vector_base<T> > rowvector = getVCLVecptr<T>(rowvectorR.slot("address"), BisVCL, ctx_id);
  std::shared_ptr<viennacl::matrix<T> > result = getVCLptr<T>(resultR.slot("address"), BisVCL, ctx_id);
  
  matrix_vector_eledivide(*matrix, *rowvector, *result, numWorkItems, ctx_id);
  
}





// [[Rcpp::export]]
void mat_vec_eledivideBackend(
    Rcpp::S4 matrixR,
    Rcpp::S4 rowvectorR,
    Rcpp::S4 resultR,
    Rcpp::IntegerVector numWorkItems) {
  
  Rcpp::traits::input_parameter< std::string >::type classVarR(RCPP_GET_CLASS(resultR));
  std::string precision_type = (std::string) classVarR;
  
  
  if(precision_type == "fvclMatrix") {
    return (mat_vec_eledivideTemplated<float>(matrixR, rowvectorR, resultR, numWorkItems));
  } else if (precision_type == "dvclMatrix") {
    return (mat_vec_eledivideTemplated<double>(matrixR, rowvectorR, resultR, numWorkItems));
  } else if (precision_type  == "ivclMatrix") {
    return( mat_vec_eledivideTemplated<int>(matrixR, rowvectorR, resultR, numWorkItems));
  }
  
}


