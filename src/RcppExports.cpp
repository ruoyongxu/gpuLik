// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// backsolveBatchBackend
SEXP backsolveBatchBackend(Rcpp::S4 C, Rcpp::S4 A, Rcpp::S4 B, Rcpp::IntegerVector Cstartend, Rcpp::IntegerVector Astartend, Rcpp::IntegerVector Bstartend, const int numbatchB, const int diagIsOne, Rcpp::IntegerVector Nglobal, Rcpp::IntegerVector Nlocal, const int NlocalCache, const int verbose);
RcppExport SEXP _gpuLik_backsolveBatchBackend(SEXP CSEXP, SEXP ASEXP, SEXP BSEXP, SEXP CstartendSEXP, SEXP AstartendSEXP, SEXP BstartendSEXP, SEXP numbatchBSEXP, SEXP diagIsOneSEXP, SEXP NglobalSEXP, SEXP NlocalSEXP, SEXP NlocalCacheSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::S4 >::type C(CSEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type A(ASEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type B(BSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type Cstartend(CstartendSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type Astartend(AstartendSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type Bstartend(BstartendSEXP);
    Rcpp::traits::input_parameter< const int >::type numbatchB(numbatchBSEXP);
    Rcpp::traits::input_parameter< const int >::type diagIsOne(diagIsOneSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type Nglobal(NglobalSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type Nlocal(NlocalSEXP);
    Rcpp::traits::input_parameter< const int >::type NlocalCache(NlocalCacheSEXP);
    Rcpp::traits::input_parameter< const int >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(backsolveBatchBackend(C, A, B, Cstartend, Astartend, Bstartend, numbatchB, diagIsOne, Nglobal, Nlocal, NlocalCache, verbose));
    return rcpp_result_gen;
END_RCPP
}
// cholBatchBackend
void cholBatchBackend(Rcpp::S4 A, Rcpp::S4 D, Rcpp::IntegerVector Astartend, Rcpp::IntegerVector Dstartend, const int numbatchD, Rcpp::IntegerVector Nglobal, Rcpp::IntegerVector Nlocal, Rcpp::IntegerVector NlocalCache);
RcppExport SEXP _gpuLik_cholBatchBackend(SEXP ASEXP, SEXP DSEXP, SEXP AstartendSEXP, SEXP DstartendSEXP, SEXP numbatchDSEXP, SEXP NglobalSEXP, SEXP NlocalSEXP, SEXP NlocalCacheSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::S4 >::type A(ASEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type D(DSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type Astartend(AstartendSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type Dstartend(DstartendSEXP);
    Rcpp::traits::input_parameter< const int >::type numbatchD(numbatchDSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type Nglobal(NglobalSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type Nlocal(NlocalSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type NlocalCache(NlocalCacheSEXP);
    cholBatchBackend(A, D, Astartend, Dstartend, numbatchD, Nglobal, Nlocal, NlocalCache);
    return R_NilValue;
END_RCPP
}
// crossprodBatchBackend
SEXP crossprodBatchBackend(Rcpp::S4 C, Rcpp::S4 A, Rcpp::S4 D, const int invertD, Rcpp::IntegerVector Cstartend, Rcpp::IntegerVector Astartend, Rcpp::IntegerVector Dstartend, Rcpp::IntegerVector Nglobal, Rcpp::IntegerVector Nlocal, const int NlocalCache, const int verbose);
RcppExport SEXP _gpuLik_crossprodBatchBackend(SEXP CSEXP, SEXP ASEXP, SEXP DSEXP, SEXP invertDSEXP, SEXP CstartendSEXP, SEXP AstartendSEXP, SEXP DstartendSEXP, SEXP NglobalSEXP, SEXP NlocalSEXP, SEXP NlocalCacheSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::S4 >::type C(CSEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type A(ASEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type D(DSEXP);
    Rcpp::traits::input_parameter< const int >::type invertD(invertDSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type Cstartend(CstartendSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type Astartend(AstartendSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type Dstartend(DstartendSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type Nglobal(NglobalSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type Nlocal(NlocalSEXP);
    Rcpp::traits::input_parameter< const int >::type NlocalCache(NlocalCacheSEXP);
    Rcpp::traits::input_parameter< const int >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(crossprodBatchBackend(C, A, D, invertD, Cstartend, Astartend, Dstartend, Nglobal, Nlocal, NlocalCache, verbose));
    return rcpp_result_gen;
END_RCPP
}
// gemmBatch2backend
SEXP gemmBatch2backend(Rcpp::S4 A, Rcpp::S4 B, Rcpp::S4 C, Rcpp::IntegerVector transposeABC, Rcpp::IntegerVector submatrixA, Rcpp::IntegerVector submatrixB, Rcpp::IntegerVector submatrixC, Rcpp::IntegerVector batches, Rcpp::IntegerVector workgroupSize, Rcpp::IntegerVector NlocalCache, const int verbose);
RcppExport SEXP _gpuLik_gemmBatch2backend(SEXP ASEXP, SEXP BSEXP, SEXP CSEXP, SEXP transposeABCSEXP, SEXP submatrixASEXP, SEXP submatrixBSEXP, SEXP submatrixCSEXP, SEXP batchesSEXP, SEXP workgroupSizeSEXP, SEXP NlocalCacheSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::S4 >::type A(ASEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type B(BSEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type C(CSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type transposeABC(transposeABCSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type submatrixA(submatrixASEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type submatrixB(submatrixBSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type submatrixC(submatrixCSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type batches(batchesSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type workgroupSize(workgroupSizeSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type NlocalCache(NlocalCacheSEXP);
    Rcpp::traits::input_parameter< const int >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(gemmBatch2backend(A, B, C, transposeABC, submatrixA, submatrixB, submatrixC, batches, workgroupSize, NlocalCache, verbose));
    return rcpp_result_gen;
END_RCPP
}
// likfitGpu_BackendP
void likfitGpu_BackendP(Rcpp::S4 yx, Rcpp::S4 coords, Rcpp::S4 params, Rcpp::S4 boxcox, Rcpp::S4 ssqY, Rcpp::S4 XVYXVX, Rcpp::S4 ssqBetahat, Rcpp::S4 detVar, Rcpp::S4 detReml, Rcpp::S4 jacobian, Rcpp::IntegerVector NparamPerIter, Rcpp::IntegerVector workgroupSize, Rcpp::IntegerVector localSize, Rcpp::IntegerVector NlocalCache, Rcpp::IntegerVector verbose, Rcpp::S4 ssqYX, Rcpp::S4 ssqYXcopy, Rcpp::S4 LinvYX, Rcpp::S4 QinvSsqYx, Rcpp::S4 cholXVXdiag, Rcpp::S4 varMat, Rcpp::S4 cholDiagMat);
RcppExport SEXP _gpuLik_likfitGpu_BackendP(SEXP yxSEXP, SEXP coordsSEXP, SEXP paramsSEXP, SEXP boxcoxSEXP, SEXP ssqYSEXP, SEXP XVYXVXSEXP, SEXP ssqBetahatSEXP, SEXP detVarSEXP, SEXP detRemlSEXP, SEXP jacobianSEXP, SEXP NparamPerIterSEXP, SEXP workgroupSizeSEXP, SEXP localSizeSEXP, SEXP NlocalCacheSEXP, SEXP verboseSEXP, SEXP ssqYXSEXP, SEXP ssqYXcopySEXP, SEXP LinvYXSEXP, SEXP QinvSsqYxSEXP, SEXP cholXVXdiagSEXP, SEXP varMatSEXP, SEXP cholDiagMatSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::S4 >::type yx(yxSEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type params(paramsSEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type boxcox(boxcoxSEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type ssqY(ssqYSEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type XVYXVX(XVYXVXSEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type ssqBetahat(ssqBetahatSEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type detVar(detVarSEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type detReml(detRemlSEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type jacobian(jacobianSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type NparamPerIter(NparamPerIterSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type workgroupSize(workgroupSizeSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type localSize(localSizeSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type NlocalCache(NlocalCacheSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type ssqYX(ssqYXSEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type ssqYXcopy(ssqYXcopySEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type LinvYX(LinvYXSEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type QinvSsqYx(QinvSsqYxSEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type cholXVXdiag(cholXVXdiagSEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type varMat(varMatSEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type cholDiagMat(cholDiagMatSEXP);
    likfitGpu_BackendP(yx, coords, params, boxcox, ssqY, XVYXVX, ssqBetahat, detVar, detReml, jacobian, NparamPerIter, workgroupSize, localSize, NlocalCache, verbose, ssqYX, ssqYXcopy, LinvYX, QinvSsqYx, cholXVXdiag, varMat, cholDiagMat);
    return R_NilValue;
END_RCPP
}
// mat_vec_eledivideBackend
void mat_vec_eledivideBackend(Rcpp::S4 matrixR, Rcpp::S4 rowvectorR, Rcpp::S4 resultR, Rcpp::IntegerVector numWorkItems);
RcppExport SEXP _gpuLik_mat_vec_eledivideBackend(SEXP matrixRSEXP, SEXP rowvectorRSEXP, SEXP resultRSEXP, SEXP numWorkItemsSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::S4 >::type matrixR(matrixRSEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type rowvectorR(rowvectorRSEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type resultR(resultRSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type numWorkItems(numWorkItemsSEXP);
    mat_vec_eledivideBackend(matrixR, rowvectorR, resultR, numWorkItems);
    return R_NilValue;
END_RCPP
}
// matrix_vector_sumBackend
void matrix_vector_sumBackend(Rcpp::S4 matrixR, Rcpp::S4 rowvectorR, Rcpp::S4 colvectorR, SEXP constantR, Rcpp::S4 sumR, Rcpp::IntegerVector numWorkItems);
RcppExport SEXP _gpuLik_matrix_vector_sumBackend(SEXP matrixRSEXP, SEXP rowvectorRSEXP, SEXP colvectorRSEXP, SEXP constantRSEXP, SEXP sumRSEXP, SEXP numWorkItemsSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::S4 >::type matrixR(matrixRSEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type rowvectorR(rowvectorRSEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type colvectorR(colvectorRSEXP);
    Rcpp::traits::input_parameter< SEXP >::type constantR(constantRSEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type sumR(sumRSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type numWorkItems(numWorkItemsSEXP);
    matrix_vector_sumBackend(matrixR, rowvectorR, colvectorR, constantR, sumR, numWorkItems);
    return R_NilValue;
END_RCPP
}
// fillParamsExtra
void fillParamsExtra(Rcpp::S4 param);
RcppExport SEXP _gpuLik_fillParamsExtra(SEXP paramSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::S4 >::type param(paramSEXP);
    fillParamsExtra(param);
    return R_NilValue;
END_RCPP
}
// maternBatchBackend
void maternBatchBackend(Rcpp::S4 var, Rcpp::S4 coords, Rcpp::S4 param, Rcpp::IntegerVector Nglobal, Rcpp::IntegerVector Nlocal, int startrow, int numberofrows, int verbose);
RcppExport SEXP _gpuLik_maternBatchBackend(SEXP varSEXP, SEXP coordsSEXP, SEXP paramSEXP, SEXP NglobalSEXP, SEXP NlocalSEXP, SEXP startrowSEXP, SEXP numberofrowsSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::S4 >::type var(varSEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type param(paramSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type Nglobal(NglobalSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type Nlocal(NlocalSEXP);
    Rcpp::traits::input_parameter< int >::type startrow(startrowSEXP);
    Rcpp::traits::input_parameter< int >::type numberofrows(numberofrowsSEXP);
    Rcpp::traits::input_parameter< int >::type verbose(verboseSEXP);
    maternBatchBackend(var, coords, param, Nglobal, Nlocal, startrow, numberofrows, verbose);
    return R_NilValue;
END_RCPP
}
// multiplyLowerDiagonalBatchBackend
SEXP multiplyLowerDiagonalBatchBackend(Rcpp::S4 output, Rcpp::S4 L, Rcpp::S4 D, Rcpp::S4 B, const int diagIsOne, std::string transformD, Rcpp::IntegerVector Nglobal, Rcpp::IntegerVector Nlocal, const int NlocalCache);
RcppExport SEXP _gpuLik_multiplyLowerDiagonalBatchBackend(SEXP outputSEXP, SEXP LSEXP, SEXP DSEXP, SEXP BSEXP, SEXP diagIsOneSEXP, SEXP transformDSEXP, SEXP NglobalSEXP, SEXP NlocalSEXP, SEXP NlocalCacheSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::S4 >::type output(outputSEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type L(LSEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type D(DSEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type B(BSEXP);
    Rcpp::traits::input_parameter< const int >::type diagIsOne(diagIsOneSEXP);
    Rcpp::traits::input_parameter< std::string >::type transformD(transformDSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type Nglobal(NglobalSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type Nlocal(NlocalSEXP);
    Rcpp::traits::input_parameter< const int >::type NlocalCache(NlocalCacheSEXP);
    rcpp_result_gen = Rcpp::wrap(multiplyLowerDiagonalBatchBackend(output, L, D, B, diagIsOne, transformD, Nglobal, Nlocal, NlocalCache));
    return rcpp_result_gen;
END_RCPP
}
// multiplyDiagonalBatchBackend
SEXP multiplyDiagonalBatchBackend(Rcpp::S4 C, Rcpp::S4 A, Rcpp::S4 B, const int inverse, Rcpp::IntegerVector Nglobal, Rcpp::IntegerVector Nlocal);
RcppExport SEXP _gpuLik_multiplyDiagonalBatchBackend(SEXP CSEXP, SEXP ASEXP, SEXP BSEXP, SEXP inverseSEXP, SEXP NglobalSEXP, SEXP NlocalSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::S4 >::type C(CSEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type A(ASEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type B(BSEXP);
    Rcpp::traits::input_parameter< const int >::type inverse(inverseSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type Nglobal(NglobalSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type Nlocal(NlocalSEXP);
    rcpp_result_gen = Rcpp::wrap(multiplyDiagonalBatchBackend(C, A, B, inverse, Nglobal, Nlocal));
    return rcpp_result_gen;
END_RCPP
}
// multiplyLowerBatchBackend
SEXP multiplyLowerBatchBackend(Rcpp::S4 C, Rcpp::S4 A, Rcpp::S4 B, const int diagIsOne, Rcpp::IntegerVector Nglobal, Rcpp::IntegerVector Nlocal, const int NlocalCache);
RcppExport SEXP _gpuLik_multiplyLowerBatchBackend(SEXP CSEXP, SEXP ASEXP, SEXP BSEXP, SEXP diagIsOneSEXP, SEXP NglobalSEXP, SEXP NlocalSEXP, SEXP NlocalCacheSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::S4 >::type C(CSEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type A(ASEXP);
    Rcpp::traits::input_parameter< Rcpp::S4 >::type B(BSEXP);
    Rcpp::traits::input_parameter< const int >::type diagIsOne(diagIsOneSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type Nglobal(NglobalSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type Nlocal(NlocalSEXP);
    Rcpp::traits::input_parameter< const int >::type NlocalCache(NlocalCacheSEXP);
    rcpp_result_gen = Rcpp::wrap(multiplyLowerBatchBackend(C, A, B, diagIsOne, Nglobal, Nlocal, NlocalCache));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_gpuLik_backsolveBatchBackend", (DL_FUNC) &_gpuLik_backsolveBatchBackend, 12},
    {"_gpuLik_cholBatchBackend", (DL_FUNC) &_gpuLik_cholBatchBackend, 8},
    {"_gpuLik_crossprodBatchBackend", (DL_FUNC) &_gpuLik_crossprodBatchBackend, 11},
    {"_gpuLik_gemmBatch2backend", (DL_FUNC) &_gpuLik_gemmBatch2backend, 11},
    {"_gpuLik_likfitGpu_BackendP", (DL_FUNC) &_gpuLik_likfitGpu_BackendP, 22},
    {"_gpuLik_mat_vec_eledivideBackend", (DL_FUNC) &_gpuLik_mat_vec_eledivideBackend, 4},
    {"_gpuLik_matrix_vector_sumBackend", (DL_FUNC) &_gpuLik_matrix_vector_sumBackend, 6},
    {"_gpuLik_fillParamsExtra", (DL_FUNC) &_gpuLik_fillParamsExtra, 1},
    {"_gpuLik_maternBatchBackend", (DL_FUNC) &_gpuLik_maternBatchBackend, 8},
    {"_gpuLik_multiplyLowerDiagonalBatchBackend", (DL_FUNC) &_gpuLik_multiplyLowerDiagonalBatchBackend, 9},
    {"_gpuLik_multiplyDiagonalBatchBackend", (DL_FUNC) &_gpuLik_multiplyDiagonalBatchBackend, 6},
    {"_gpuLik_multiplyLowerBatchBackend", (DL_FUNC) &_gpuLik_multiplyLowerBatchBackend, 7},
    {NULL, NULL, 0}
};

RcppExport void R_init_gpuLik(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
