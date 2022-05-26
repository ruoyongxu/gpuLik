#include <string>

#define GSL_FLT_EPSILON 1.1920928955078125e-07
#define GSL_DBL_EPSILON 2.2204460492503131e-16
//#define GSL_SQRT_DBL_MAX 1.3407807929942596e+154

template <typename T> 
int sizeOfReal() {
  return(-1);}

template <> int sizeOfReal<double>(){
  return(sizeof(double));}

template <> int sizeOfReal<float>(){
  return(sizeof(float));}



template <typename T> 
std::string openclTypeString() {
  return("undefined");
}

template <> 
std::string openclTypeString<double>(){
  std::string result = "double";
  return(result);
}

template <> 
std::string openclTypeString<float>(){
  std::string result = "float";
  return(result);
}

template <>
std::string openclTypeString<uint>(){
  std::string result = "uint";
  return(result);
}

template <>
std::string openclTypeString<int>(){
  std::string result = "int";
  return(result);
}


template <typename T> 
T maternClEpsilon(){
  return(-1);
}

template <> 
double maternClEpsilon<double>(){
  return(GSL_DBL_EPSILON);
}

template <> 
float maternClEpsilon<float>(){
  return(GSL_FLT_EPSILON);
}

















