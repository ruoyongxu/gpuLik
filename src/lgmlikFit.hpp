
#include <Rcpp.h>
#include <string>

#include <dynMatrix/dynVCLMatGeostatsgpu.hpp>
#include <dynMatrix/dynVCLVecGeostatsgpu.hpp>
#include "viennacl/linalg/sum.hpp"
#include "viennacl/ocl/backend.hpp"
#include <Rmath.h>




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
int logDet);

template <typename T> 
std::string backsolveBatchString(
    const int sameB,
    const int diagIsOne,
    const int Nrow, 
    const int Ncol,
    //    const int Nmatrix, 
    const int NpadC, 
    const int NpadA, 
    const int NpadB,
    const int NpadBetweenMatricesC,
    const int NpadBetweenMatricesA,
    const int NpadBetweenMatricesB,
    const int NstartC,
    const int NstartA,
    const int NstartB,
    const int NrowsToCache, 
    const int NcolsPerGroup,
    const int NlocalCacheC,  // NrowsToCache by NcolsPerGroup
    const int NlocalCacheSum, // Nlocal(0) * Nlocal(1) * NcolsPerGroup 
    const int NpadBetweenMatricesSum // Nlocal(0) * Nlocal(1)
);

template <typename T> std::string crossprodBatchString(
    const int Nrow, 
    const int Ncol,
    //    const int Nmatrix, 
    const int NpadC, 
    const int NpadA,
    const int NpadD, // set to zero to omit D
    const int invertD, // set to 1 for A^T D^(-1) A
    const int onlyDiagC,
    const int NstartC,  // newly added
    const int NstartA,  // new
    const int NstartD,  // new
    const int NpadBetweenMatricesC,
    const int NpadBetweenMatricesA,
    const int NlocalCacheA // numbers of rows to cache of A.  Rcpp::IntegerVector Nlocal// cache a Nlocal[0] by Nlocal[1] submatrix of C
);

template<typename T> void fill22params(
    viennacl::matrix_base<T> &param
); 


template <typename T> std::string openclTypeString();


template <typename T> 
std::string gemmBatch2String(
    const int onlyDiagC,         // compute only the diagonal values of C
    Rcpp::IntegerVector transposeABC,  
    Rcpp::IntegerVector submatrixA,  
    Rcpp::IntegerVector submatrixB,
    Rcpp::IntegerVector submatrixC,
    Rcpp::IntegerVector recycle,      //"nCol", "recycleArow", "recycleAcol", "recycleBrow", "recycleBcol"
    Rcpp::IntegerVector NlocalCache,
    int NpadA, int NpadB, int NpadC);






//////////////////////////////////////////////////////////////////////////////////////////////////////
//#include "viennacl/linalg/lu.hpp"

extern "C" void Rtemme_gamma(double *nu, double * g_1pnu, double * g_1mnu, double *g1, double *g2);

//template<typename T> double luT(viennacl::matrix<T> &vclX, viennacl::vector_base<T> &vclD);

template <typename T> T maternClEpsilon();




#include "viennacl/vector_proxy.hpp"
#include "viennacl/matrix_proxy.hpp"




template <typename T> 
std::string maternBatchKernelString(
    int maxIter, 
    int N,
    int Ncell,
    int Npad,
    int Nmatrix,
    int NpadBetweenMatrices,
    int NpadCoords,
    int NpadParams,
    int Nlocal0,
//    int NlocalParamsCache,
    int assignUpper,
    int assignLower,
    int assignDiagonals,
    int assignDistUpper);

template<typename T> 
void maternBatchVcl(
    viennacl::matrix_base<T> &vclVar, // Nmat columns N^2 rows
    viennacl::matrix_base<T> &vclCoords, // 2 columns
    viennacl::matrix_base<T> &param, // Nmat rows, 22 columns
    Rcpp::IntegerVector numWorkItems,
    Rcpp::IntegerVector numLocalItems,	
    const int ctx_id,
    int startrow,   // new added
    int numberofrows,
    int verbose);
    
    
    template <typename T> 
    int cholBatchVcl(
        viennacl::matrix<T> &A,
        viennacl::matrix<T> &D,
        Rcpp::IntegerVector Astartend,
        Rcpp::IntegerVector Dstartend,  
        const int numbatchD,
        Rcpp::IntegerVector Nglobal,
        Rcpp::IntegerVector Nlocal, 
        Rcpp::IntegerVector NlocalCache,
        const int ctx_id);
    
    
    // template <typename T>
    // void rowsum(
    //     viennacl::matrix<T> &x,
    //     viennacl::vector_base<T> &Sum,
    //     std::string type,
    //     int log);
        
        
        template <typename T> 
        void backsolveBatch(
            viennacl::matrix<T> &C,
            viennacl::matrix<T> &A,  //must be batches of square matrices
            viennacl::matrix<T> &B,
            Rcpp::IntegerVector Cstartend,
            Rcpp::IntegerVector Astartend, //square matrices
            Rcpp::IntegerVector Bstartend, 
            const int numbatchB,
            const int diagIsOne,
            Rcpp::IntegerVector Nglobal,
            Rcpp::IntegerVector Nlocal,
            const int NlocalCache,
            const int verbose,
            const int ctx_id);    
        
        
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
            const int NlocalCache, //int NrowStartC,
            const int ctx_id);     
      
        
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
            const int ctx_id); 
        
        
        
// template<typename T> 
// void matrix_vector_sum(
//      viennacl::matrix<T> &matrix,// viennacl::vector_base<int>  rowSum, viennacl::vector_base<int>  colSum,  
//      viennacl::vector_base<T> &vector,
//      viennacl::matrix<T> &sum,
//      const int byrow,
//      Rcpp::IntegerVector numWorkItems,
//      const int ctx_id);            
//                 
//                 
//                 template<typename T> 
//                 void matrix_scalar_sum(
//                     viennacl::matrix<T> &matrix,// viennacl::vector_base<int>  rowSum, viennacl::vector_base<int>  colSum,  
//                     T  value,         //viennacl::scalar<T> value,     
//                     viennacl::matrix<T> &sum,
//                     Rcpp::IntegerVector numWorkItems,
//                     const int ctx_id);                
                
                
                
                
                
        