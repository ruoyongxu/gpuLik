#include "lgmlikFit.hpp"
//#define DEBUG


/*
 solve L a = b for a
 solve A C = B for C
 
 
 
 - Dglobal[0] is matrix, Dlocal[0] is row of A and C
 - Dglobal[1] is column of B, Dlocal[1] is Dinner
 - each Dglobal[1] does NcolsGroup columns of B
 - NcolsPerGroup = ceil( Ncol/Nglobal[1])
 - local cache Nlocal[1] submatrices of size Nlocal[0] by NcolsPerGroup of C
 - don't cache A ?
 
 - loop Dmatrix = Dglobal[0]
 - loop Drow = Dlocal[0]
 - loop Dinner = Dlocal[1]
 - cache A ?
 - loop DcolB = Dglobal[1]
 - compute part of Linv b
 - end Dinner, DcolB, 
 - sum Linv b over Dlocal[1] in cache
 - the Dlocal[1]=0's fill in a's
 
 
 
 */

//  solve A C = B for C, A must be lower triangular 

template <typename T> 
std::string backsolveBatchString(
    const int sameB,   //1
    const int diagIsOne,
    const int Nrow,    //3
    const int Ncol,
    //    const int Nmatrix, 
    const int NpadC,   //5
    const int NpadA,   
    const int NpadB,   //7
    const int NpadBetweenMatricesC, 
    const int NpadBetweenMatricesA, //9
    const int NpadBetweenMatricesB,
    const int NstartC,  //11
    const int NstartA,  
    const int NstartB,  //13
    const int NrowsToCache,  
    const int NcolsPerGroup,  //15
    const int NlocalCacheC,  // NrowsToCache by NcolsPerGroup 
    const int NlocalCacheSum, // Nlocal(0) * Nlocal(1) * NcolsPerGroup 17
    const int NpadBetweenMatricesSum // Nlocal(0) * Nlocal(1)  18
) {
  
  std::string typeString = openclTypeString<T>();
  std::string result = "";
  
  if(typeString == "double") {
    result += "\n#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\n";
  }
  result = result + 
    "\n#define Nrow " + std::to_string(Nrow) + "\n"    
    "#define Ncol " + std::to_string(Ncol) + "\n"    
    //    "#define Nmatrix " + std::to_string(Nmatrix) + "\n"    
    "#define NpadC " + std::to_string(NpadC) + "\n"
    "#define NpadA " + std::to_string(NpadA) + "\n"    
    "#define NpadB " + std::to_string(NpadB) + "\n"
    "#define NpadCcache " + std::to_string(NcolsPerGroup) + "\n"
    "#define NpadBetweenMatricesC " + std::to_string(NpadBetweenMatricesC) + "\n"    
    "#define NpadBetweenMatricesA " + std::to_string(NpadBetweenMatricesA) + "\n"    
    "#define NpadBetweenMatricesB " + std::to_string(NpadBetweenMatricesB) + "\n"   
    "#define NstartC " + std::to_string(NstartC) + "\n"   
    "#define NstartA " + std::to_string(NstartA) + "\n"   
    "#define NstartB " + std::to_string(NstartB) + "\n"   
    "#define NrowsToCache " + std::to_string(NrowsToCache) + "\n"
    "// min(Nrow, NrowsToCache)\n"
    "#define NrowStop " + std::to_string(std::min(Nrow, NrowsToCache)) + "\n"
    "// NrowsToCache by NcolsPerGroup\n"
    "#define NlocalCacheC " + std::to_string(NlocalCacheC) + "\n"    
    "// Nlocal(0) * Nlocal(1) * NcolsPerGroup \n"
    "#define NlocalCacheSum " + std::to_string(NlocalCacheSum) + "\n"
    "#define NpadBetweenMatricesSum " + std::to_string(NpadBetweenMatricesSum) + "\n\n";
  
  result += "__kernel void backsolveBatch(\n"
  " __global " + typeString+ " *C,\n"
  " __global "+ typeString+ " *A,\n"
  " __global "+ typeString+ " *B,\n"
  " __local " + typeString + " *localCache,\n"
  "           int Nmatrix) {\n\n"
  "local int AHere, BHere, CHere, DrowZero;\n"
  "int AHereRow, BHereRow, CHereRow, CcacheHereRow;\n"
  + typeString + " Acache;\n" 
  "int DmatrixBlock, Dmatrix, DmatrixInBounds, Drow, DrowBlock, DrowInBounds;\n"
  "int Dcol,DcolBlock, DcolInBounds, Dinner, DinnerBlock, DinnerC, DcolCache, Dsum;\n"
  "local int DCrowInc, DArowInc, DBrowInc, DinnerCinc, DCcacheRowInc;\n"
  "const int DlocalCache = get_local_id(0) * get_local_size(1) + get_local_id(1);\n"
  "const int localIsFirstCol = (get_local_id(1) == 0);\n"
  "const int localIsFirstItem = (get_local_id(0) == 0) & (get_local_id(1) == 0);\n";
  
  result +=   
    "/*\n"
    " * local cache, size of localCache must exceed NlocalChaceC + NlocalCacheSum\n"
    " */\n"
    "local "+ typeString+ " *Ccache = localCache;\n" 
    "local "+ typeString+ " *cacheSum = &localCache[NlocalCacheC];\n";
  
  result += "\n";
    
  result += 
    "if (localIsFirstItem) {\n"
    "  DCrowInc = get_local_size(0) * NpadC;\n"
    "  DCcacheRowInc = get_local_size(0) * NpadCcache;\n"
    "  DArowInc = get_local_size(0) * NpadA;\n"
    "  DBrowInc = get_local_size(0) * NpadB;\n"
    //    "  DinnerCinc = get_local_size(1) * NpadC;\n"
    "  DinnerCinc = get_local_size(1) * NpadCcache;\n"
    "}\n";
  // loop through matrix
  result +=  "\n"
  // "for(Dmatrix = get_group_id(0); Dmatrix < Nmatrix; Dmatrix += get_num_groups(1)){\n"
  
  "for(Dmatrix=get_group_id(0);\n "
  "    Dmatrix < Nmatrix;\n "
  "    Dmatrix += get_num_groups(0)){\n"
  

  "     DmatrixInBounds = 1;\n";
  
  
result +=
  "  if(localIsFirstItem) {\n"
  "    AHere = Dmatrix*NpadBetweenMatricesA + NstartA;\n"
  "    CHere = Dmatrix*NpadBetweenMatricesC + NstartC;\n";
  if(sameB){
    result +=  "    BHere = NstartB;\n";
  } else {
    result +=  "    BHere = Dmatrix*NpadBetweenMatricesB + NstartB;\n";
  }
  result +=  "  };\n"
  "  barrier(CLK_LOCAL_MEM_FENCE);\n\n";

  
  // loop through rows of A
  result +=
    "   for(DrowBlock = 0,\n"
    "         BHereRow = BHere + get_local_id(0) * NpadB,\n"
    "         CHereRow = CHere + get_local_id(0) * NpadC,\n"
    "         CcacheHereRow = get_local_id(0) * NpadCcache,\n"
    "         AHereRow = AHere + get_local_id(0) * NpadA;\n"
    "       DrowBlock < NrowStop;\n"
    "       DrowBlock += get_local_size(0), \n"
    "         BHereRow += DBrowInc,\n"
    "         AHereRow += DArowInc,\n "
    "         CHereRow += DCrowInc,\n"
    "         CcacheHereRow += DCcacheRowInc){\n"
    
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"    
    "    Drow = DrowBlock + get_local_id(0);\n";
  
  
  result +=
    "    if((Drow < NrowStop)  & DmatrixInBounds){\n"
    "       DrowInBounds = 1;\n"
    "    } else {\n"
    "       DrowInBounds = 0;\n"
    "       Drow = 0;\n"
    "    }\n"
    "    if(localIsFirstItem) {\n"
    "      DrowZero = Drow;\n"
    "    };\n";
  



  result += 
    "    // initialize cacheSum\n"

    "    for(Dcol = get_group_id(1), DcolCache=DlocalCache;\n"
    "      Dcol < Ncol;\n"
    "      Dcol += get_num_groups(1), DcolCache+=NpadBetweenMatricesSum){\n";
  
  
  result += 
    "        cacheSum[DcolCache] = 0;\n";
  
  result += 
    "    }//for Dcol\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n";
  //#endif
  
  
  
  //  #ifdef UNDEF 
//  result += 
 //   "     for(DinnerBlock = 0;\n"
  //  "         DinnerBlock < DrowZero;\n"
   // "         DinnerBlock += get_local_size(1)){\n";

  result +=
    "    if(DrowInBounds){\n";
     result += 
    "    for(Dinner = get_local_id(1),\n"
    "      DinnerC =Dinner*NpadCcache;\n"
    "      Dinner < DrowZero;\n"
    "      Dinner += get_local_size(1), DinnerC += DinnerCinc){\n";
     
//  result +=
//    "         Dinner = DinnerBlock + get_local_id(1);\n";
//  result +=
 //   "         DinnerC = Dinner*NpadCcache;\n";
  result += 
    "      Acache = A[AHereRow + Dinner];\n"; 
  //"      Acache = A[AHere + Drow*NpadA + Dinner];\n"; 
  result += 
    "      // loop through columns of B and C\n"
     "      for(Dcol = get_group_id(1),DcolCache=0, Dsum=DlocalCache;\n"
     "        Dcol < Ncol;\n"
     "        Dcol += get_num_groups(1), DcolCache++, Dsum+=NpadBetweenMatricesSum){\n";
    //"    for(DcolBlock = 0, DcolCache=0, Dsum=DlocalCache;\n"
    //"        DcolBlock < Ncol;\n"
    //"        DcolBlock += get_num_groups(1), DcolCache++, Dsum+=NpadBetweenMatricesSum){\n"
    //"        Dcol = DcolBlock + get_group_id(1);\n";
//  result +=   
//    "        if(DrowInBounds) {\n";
  result += 
    "          cacheSum[Dsum] += \n"
    //      "        cacheSum[DlocalCache + DcolCache * NpadBetweenMatricesSum] +=\n"
    "            Acache * Ccache[DinnerC+ DcolCache];\n";
  //    "           Acache * Ccache[Dinner * NpadCcache + DcolCache];\n";
  result += 
    "      }//for Dcol\n";

  
  result += 
    "    }//for Dinner\n";
  
  result += 
    "    }//if DrowInBounds\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n";
  //  #endif 
  
  
  result += 
    "    // loop columns again\n"
    "    for(Dcol = get_group_id(1),DcolCache=0;\n"
    "      Dcol < Ncol;\n"
    "      Dcol += get_num_groups(1), DcolCache++){\n";
  
  
  result +=
    "        if(DrowInBounds & localIsFirstCol){\n";  
  
  
    result += 
    "        for(Dinner = 1, DinnerC = DlocalCache+1;\n"
    "            Dinner < get_local_size(1);\n"
    "            Dinner++,DinnerC++){\n";
  
  result +=
    "            cacheSum[NpadBetweenMatricesSum*DcolCache + DlocalCache] +=\n"
    "              cacheSum[NpadBetweenMatricesSum*DcolCache + DinnerC];\n"; 
  
  result +=
    "        }//Dinner\n"
    "        }// DrowInBounds & localIsFirstCol\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n";
  
    result +=
    "       // last bit of the triangle\n"    
    "       for(Dinner = 0,DinnerC = 0;\n"
    "         Dinner < get_local_size(0);\n"
    "         Dinner++,DinnerC += get_local_size(1)){\n";  
  
  result +=
    "        // create C in cache and copy to C\n"
    "        if( DrowInBounds & localIsFirstCol){\n"
    "        if((get_local_id(0) == Dinner)){\n"   

    "          cacheSum[NpadBetweenMatricesSum*DcolCache + DinnerC] = (B[BHereRow + Dcol] -\n"
    "               cacheSum[NpadBetweenMatricesSum*DcolCache + DinnerC])";
  
  if(!diagIsOne){
    result += "/ A[AHere + Drow * NpadA + Drow]";
  }
  result += ";\n";  
  result +=
    "          Ccache[CcacheHereRow + DcolCache] =\n"
    "            cacheSum[NpadBetweenMatricesSum*DcolCache + DinnerC];\n"; 
  result +=  
    "          C[CHereRow + Dcol] = Ccache[CcacheHereRow + DcolCache];\n";  


  result +=  
    "        } //if(get_local_id(0) == Dinner)\n"
    "        }// if DrowInBounds & localIsFirstCol\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n";  
  result +=
    "        // update A[Drow, ] * B[, Dcol] \n"
    "        if( DrowInBounds & localIsFirstCol){\n"    
    "        if((get_local_id(0) > Dinner)){\n"
    "          cacheSum[NpadBetweenMatricesSum*DcolCache + DlocalCache] +=\n"
    "            A[AHereRow + DrowZero + Dinner] * cacheSum[NpadBetweenMatricesSum*DcolCache + DinnerC];\n"
    "        }//if(get_local_id(0) > Dinner)\n"
    "        }// if DrowInBounds & localIsFirstCol\n"    
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"; 
  
  result +=
    "        }//Dinner\n";
  result +=
    "      barrier(CLK_LOCAL_MEM_FENCE);\n";  
  result += 
    "    }//for DcolBlock\n";


  result +=
    "  }//for Drow\n\n\n";
  
  
  
  
  

  result += 
    "/*\n Now rows that aren't all cached\n*/\n";
  result +=
    // "  for(Drow = NrowStop + get_local_id(0),\n"
    // "      BHereRow = BHere + Drow * NpadB,\n"
    // "      CHereRow = CHere + Drow * NpadC,\n"
    // // "      CcacheHereRow = Drow * NpadCcache,\n"
    // "      CcacheHereRow = get_local_id(0) * NpadCcache,\n"
    // "      AHereRow = AHere + Drow * NpadA;\n"
    // "      Drow < Nrow;\n"
    // "      Drow += get_local_size(0), \n"
    // "      BHereRow += DBrowInc,\n"
    // "      AHereRow += DArowInc, CHereRow += DCrowInc,\n"
    // "      CcacheHereRow += DCcacheRowInc){\n"
    
    "    for(DrowBlock = NrowStop;\n"
    "        DrowBlock < Nrow;\n"
    "        DrowBlock += get_local_size(0)){\n"
    
    "      Drow = DrowBlock + get_local_id(0);\n"   
    "      if((Drow < Nrow) & DmatrixInBounds){\n"
    "        DrowInBounds = 1;\n"
    "      } else {\n"
    "        DrowInBounds = 0;\n"
    "        Drow = 0;\n"
    "      }\n"
    
    "        BHereRow = BHere + Drow * NpadB,\n"
    "        CHereRow = CHere + Drow * NpadC,\n"
    "        CcacheHereRow = get_local_id(0) * NpadCcache,\n"
    "        AHereRow = AHere + Drow * NpadA;\n"    
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    
    
    "    if(localIsFirstItem) {\n"
    "      DrowZero = Drow;\n"
    "    };\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n";
  
  result += 
    "    // initialize cacheSum\n"
    "        if(DrowInBounds){\n"
    "    for(Dcol = get_group_id(1), DcolCache=DlocalCache;\n"
    "        Dcol < Ncol;\n"
    "        Dcol += get_num_groups(1), DcolCache+=NpadBetweenMatricesSum){\n";
    // "    for(DcolBlock = 0, DcolCache=DlocalCache;\n"
    // "        DcolBlock < Ncol;\n"
    // "        DcolBlock += get_num_groups(1), DcolCache+=NpadBetweenMatricesSum){\n"
    // "        Dcol = DcolBlock + get_group_id(1);\n"   
    // "        if((Dcol < Ncol) & DrowInBounds)\n";     
  result += 
    "        cacheSum[DcolCache] = 0.0;\n";
  result += 
    "    }//for Dcol\n"
    "    }// if\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n";
  
  result += 
    // "    for(Dinner = get_local_id(1),\n"
    // "      DinnerC = get_local_id(1)*NpadCcache;\n"
    // "      Dinner < NrowStop;\n"
    // "      Dinner += get_local_size(1), DinnerC += DinnerCinc){\n";
    
    "for(DinnerBlock = 0,\n"
    "    DinnerC = get_local_id(1)*NpadCcache;\n"
    "    DinnerBlock < NrowStop;\n"
    "    DinnerBlock += get_local_size(1), DinnerC += DinnerCinc){\n"
    "    Dinner = DinnerBlock + get_local_id(1);\n"
    "    if((Dinner < NrowStop) & DrowInBounds){\n";        
  
  result += 
    "      Acache = A[AHereRow + Dinner];\n"; 
  //"      Acache = A[AHere + Drow*NpadA + Dinner];\n"; 
  
  result += 
    "      // loop through columns of B and C\n"
    "      for(Dcol = get_group_id(1),DcolCache=0, Dsum=DlocalCache;\n"
    "        Dcol < Ncol;\n"
    "        Dcol += get_num_groups(1), DcolCache++, Dsum+=NpadBetweenMatricesSum){\n";
    // "    for(DcolBlock = 0, DcolCache=0, Dsum=DlocalCache;\n"
    // "        DcolBlock < Ncol;\n"
    // "        DcolBlock += get_num_groups(1), DcolCache++, Dsum+=NpadBetweenMatricesSum){\n"
    // "        Dcol = DcolBlock + get_group_id(1);\n"      
    // "        if(Dcol < Ncol)\n";
  
  result +=     
    "        cacheSum[Dsum] += Acache * Ccache[DinnerC + DcolCache];\n";
  //    "        cacheSum[DlocalCache + DcolCache * NpadBetweenMatricesSum] +=\n"
  //  "           Acache * Ccache[Dinner * NpadCcache + DcolCache];\n"; 
  result += 
    "      }//for Dcol\n"
    "       }// if\n";
  result += 
    "    }//for Dinner\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n";
  
  result += 
    // "    for(Dinner = NrowStop + get_local_id(1),\n"
    // "      DinnerC = Dinner*NpadCcache;\n"
    // "      Dinner < DrowZero;\n"
    // "      Dinner += get_local_size(1), DinnerC += DinnerCinc){\n";
    "    for(DinnerBlock = NrowStop;\n"
    "        DinnerBlock < DrowZero;\n"
    "        DinnerBlock += get_local_size(1)){\n"    
    "        Dinner = DinnerBlock + get_local_id(1);\n"
    "        if((Dinner < DrowZero) & DrowInBounds){\n";
  result += 
    "      DinnerC = Dinner*NpadCcache;\n"
    "      Acache = A[AHereRow + Dinner];\n"; 
  //"      Acache = A[AHere + Drow*NpadA + Dinner];\n";  
  result += 
    "      // loop through columns of B and C\n"
    "      for(Dcol = get_group_id(1),DcolCache=0, Dsum=DlocalCache;\n"
    "        Dcol < Ncol;\n"
    "        Dcol += get_num_groups(1), DcolCache++, Dsum+=NpadBetweenMatricesSum){\n";
    // "    for(DcolBlock = 0, DcolCache=0, Dsum=DlocalCache;\n"
    // "        DcolBlock < Ncol;\n"
    // "        DcolBlock += get_num_groups(1), DcolCache++, Dsum+=NpadBetweenMatricesSum){\n"
    // "        Dcol = DcolBlock + get_group_id(1);\n"      
    // "        if(Dcol < Ncol)\n";
  result += 
    "        cacheSum[Dsum] += Acache * C[CHere + Dcol + NpadC * Dinner];\n";
  //    "        cacheSum[DlocalCache + DcolCache * NpadBetweenMatricesSum] +=\n"
  //  "           Acache * Ccache[Dinner * NpadCcache + DcolCache];\n";
  result += 
    "      }//for Dcol\n"
    "      }// if\n"; 
  result += 
    "    }//for Dinner\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n";
  //#endif  
  
  
  result += 
    "    // loop columns again\n"
    "    for(Dcol = get_group_id(1),DcolCache=0;\n"
    "      Dcol < Ncol;\n"
    "      Dcol += get_num_groups(1), DcolCache++){\n";
    // "    for(DcolBlock = 0, DcolCache=0;\n"
    // "        DcolBlock < Ncol;\n"
    // "        DcolBlock += get_num_groups(1), DcolCache++){\n"
    // "        Dcol = DcolBlock + get_group_id(1);\n";      
  
  result += "\n"
  "      if(localIsFirstCol & DrowInBounds){\n";
  result += 
    "        for(Dinner = 1, DinnerC = DlocalCache+1;\n"
    "            Dinner < get_local_size(1);\n"
    "            Dinner++,DinnerC++){\n";
  result += 
    "            cacheSum[NpadBetweenMatricesSum*DcolCache + DlocalCache] +=\n"
    "              cacheSum[NpadBetweenMatricesSum*DcolCache + DinnerC];\n";
  
  result += 
    "        }//Dinner\n"
    "        }// if DrowInBounds & localIsFirstCol\n"    
    "        barrier(CLK_LOCAL_MEM_FENCE);\n";
  
  result +=
    "       // last bit of the triangle\n"    
    "       for(Dinner = 0,DinnerC = 0;\n"
    "           Dinner < get_local_size(0);\n"
    "           Dinner++,DinnerC += get_local_size(1)){\n";
  
  result +=
    "        // create C in cache and copy to C\n"
    "      if(localIsFirstCol & DrowInBounds){\n"
    "          if(get_local_id(0) == Dinner){\n"  
    "          cacheSum[NpadBetweenMatricesSum*DcolCache + DinnerC] = (B[BHereRow + Dcol] -\n"
    "           cacheSum[NpadBetweenMatricesSum*DcolCache + DinnerC])";
  if(!diagIsOne){
    result += "/ A[AHere + Drow * NpadA + Drow]";
  }
  result += ";\n";
  // DEBUGGING!!
  result +=
    "      C[CHereRow + Dcol] = cacheSum[NpadBetweenMatricesSum*DcolCache + DinnerC];\n";
  //      "          C[CHereRow + Dcol] = 10*(1 + Dmatrix) + (1+Drow) + (1+Dcol)/10;\n";
  
  result +=
    "        } //if(get_local_id(0) == Dinner)\n"
    "        }// if DrowInBounds & localIsFirstCol\n"       
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"; 
  result +=
    "        // update A[Drow, ] * B[, Dcol] \n"
    "      if(localIsFirstCol & DrowInBounds){\n"
    "        if(get_local_id(0) > Dinner){\n"
    "          cacheSum[NpadBetweenMatricesSum*DcolCache + DlocalCache] +=\n"
    "            A[AHereRow + DrowZero + Dinner] * cacheSum[NpadBetweenMatricesSum*DcolCache + DinnerC];\n"
    "        }//if(get_local_id(0) > Dinner)\n"
    "        }//if DrowInBounds & localIsFirstCol\n"  
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"; 
  
  
  result +=
    "        }//Dinner\n";
  result +=
    "      barrier(CLK_LOCAL_MEM_FENCE);\n";  
  result += 
    "    }//for DcolBlock\n";
  result += 
    "  }//for Drow\n\n";

  result += 
    "}// Dmatrix\n";
  
  result += 
    "}// kernel\n";
  
  /*
   * TO DO: do remaining rows
   * multiply A with cached C
   *   copy C over
   * loop through cached blocks of C
   * do triangle bit
   *    
   */
  return(result);
}



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
    const int ctx_id) {
  
  
  
  const int NstartC = C.internal_size2() * Cstartend[0] + Cstartend[2];
  const int NstartA = A.internal_size2() * Astartend[0] + Astartend[2];
  const int NstartB = B.internal_size2() * Bstartend[0] + Bstartend[2];
  const int Nrow = Astartend[1], Ncol = Bstartend[3];
  const int Nmatrix = A.size1()/A.size2();
  //const int fullCrow = C.size1/Nmatrix;
  //const int fullBrow = B.size1/numbatchB;
  
  const int Ngroups1 = static_cast<T>(Nglobal[1]) / static_cast<T>(Nlocal[1]);
  const int NcolsPerGroup = std::ceil( static_cast<T>(Ncol) / static_cast<T>(Ngroups1));
  const int NlocalCacheSum = Nlocal[0] * Nlocal[1] * NcolsPerGroup;
  
  const int NrowsToCache = std::floor(static_cast<T>(NlocalCache - NlocalCacheSum-1) / static_cast<T>(NcolsPerGroup));
  const int NlocalCacheC = NcolsPerGroup * NrowsToCache;
  
  
  viennacl::ocl::local_mem localCache(NlocalCache*sizeof(A(0,0) ) );
  
  if(verbose) {
    
    Rcpp::Rcout << "\nNrow " << Nrow  << " Nmatrix " << Nmatrix << " Ncol " << Ncol << " NlocalCache " << NlocalCache << "\n\n";
    
  }  
  
  // the context
  viennacl::ocl::context ctx(viennacl::ocl::get_context(ctx_id));
  
  //  cl_device_type type_check = ctx.current_device().type();
  
  
  std::string clString =  backsolveBatchString<T>(  
    numbatchB==1,        //Nrow == B.size1(),    //1
    diagIsOne,  //2
    Nrow, //3
    Ncol, // ncol   4
    //    Nmatrix,  
    C.internal_size2(),     //5
    A.internal_size2(),     //6
    B.internal_size2(),     //7
    C.internal_size2()*C.size1()/Nmatrix,//NpadBetweenMatricesC,  8
    A.internal_size2()*A.size2(),//NpadBetweenMatricesA,          9
    B.internal_size2()*B.size1()/numbatchB,//NpadBetweenMatricesB,  10
    NstartC,        // 11
    NstartA,        // 12
    NstartB,        // 13
    NrowsToCache,   // 14
    NcolsPerGroup,  // 15
    NlocalCacheC,   // 16
    NlocalCacheSum, // 17
    Nlocal[0] * Nlocal[1] //  18
  );  //NcolsInCache
  
  if(verbose > 2) {
    
    Rcpp::Rcout << clString << "\n\n";
    
  }  
  
  
  viennacl::ocl::program & my_prog = ctx.add_program(clString, "my_kernel");
  
  viennacl::ocl::kernel & backsolveKernel = my_prog.get_kernel("backsolveBatch");
  
  backsolveKernel.global_work_size(0, Nglobal[0]);
  backsolveKernel.global_work_size(1, Nglobal[1]);
  
  backsolveKernel.local_work_size(0, Nlocal[0]);
  backsolveKernel.local_work_size(1, Nlocal[1]);
  
  viennacl::ocl::command_queue theQueue = backsolveKernel.context().get_queue();
  viennacl::ocl::enqueue(backsolveKernel(C, A, B, localCache, Nmatrix));
  clFinish(theQueue.handle().get());
  
}






template <typename T> 
SEXP backsolveBatchTyped(
    Rcpp::S4 CR,
    Rcpp::S4 AR,
    Rcpp::S4 BR,
    Rcpp::IntegerVector Cstartend,
    Rcpp::IntegerVector Astartend,
    Rcpp::IntegerVector Bstartend,
    const int numbatchB,
    const int diagIsOne,
    Rcpp::IntegerVector Nglobal,
    Rcpp::IntegerVector Nlocal, 
    const int NlocalCache,
    const int verbose) {
  /*
   std::vector<int> Nglobal = Rcpp::as<std::vector<int> >(NglobalR);
   std::vector<int> Nlocal = Rcpp::as<std::vector<int> >(NlocalR);*/
  
  const int ctx_id = INTEGER(CR.slot(".context_index"))[0]-1;
  const bool BisVCL=1;
  
  
  
  std::shared_ptr<viennacl::matrix<T> > AG = getVCLptr<T>(AR.slot("address"), BisVCL, ctx_id);
  std::shared_ptr<viennacl::matrix<T> > BG = getVCLptr<T>(BR.slot("address"), BisVCL, ctx_id);
  std::shared_ptr<viennacl::matrix<T> > CG = getVCLptr<T>(CR.slot("address"), BisVCL, ctx_id);
  
  backsolveBatch<T>(*CG, *AG, *BG, 
                    Cstartend, Astartend, Bstartend,
                    numbatchB,diagIsOne, 
                    Nglobal, Nlocal, NlocalCache, verbose,
                    ctx_id);
  
  return Rcpp::wrap(0L);
  
}

//if diagIsOne=TRUE, then backsolve ignores the diagonal entries in A and assumes they're all 1.  
//if diagIsOne=FALSE then backsolve will use the diagonal elements provided in A.

// [[Rcpp::export]]
SEXP backsolveBatchBackend(
    Rcpp::S4 C,
    Rcpp::S4 A,
    Rcpp::S4 B,
    Rcpp::IntegerVector Cstartend,
    Rcpp::IntegerVector Astartend,
    Rcpp::IntegerVector Bstartend,
    const int numbatchB,
    const int diagIsOne,
    Rcpp::IntegerVector Nglobal,
    Rcpp::IntegerVector Nlocal, 
    const int NlocalCache = 1000L,
    const int verbose =0L) {
  
  SEXP result;
  
  Rcpp::traits::input_parameter< std::string >::type  classVarR(RCPP_GET_CLASS(C));
  std::string precision_type = (std::string) classVarR;
  
  if(precision_type == "fvclMatrix") {
    result = backsolveBatchTyped<float>(
      C, A, B, Cstartend, Astartend, Bstartend,
      numbatchB, diagIsOne, Nglobal, Nlocal, NlocalCache, verbose);
  } else if (precision_type == "dvclMatrix") {
    result = backsolveBatchTyped<double>(
      C, A, B, Cstartend, Astartend, Bstartend,
      numbatchB, diagIsOne, Nglobal, Nlocal, NlocalCache, verbose);
  } else {
    result = Rcpp::wrap(1L);
  }
  return(result);
  
}



















