// Author: Hong Hung Sept 2006-2012

 struct _float4
 {
  float x,y,z,w;
 };
 typedef struct _float4 float4;
 struct _float3
 {
  float x,y,z;
 };
 typedef struct _float3 float3;
 struct _float2
 {
  float x,y;
 };
 typedef struct _float2 float2;
 struct _int4
 {
  int x,y,z,w;
 };
 typedef struct _int4 int4;
 struct _int3
 {
  int x,y,z;
 };
 typedef struct _int3 int3;
 struct _int2
 {
  int x,y;
 };
 typedef struct _int2 int2;

#ifdef __cplusplus
extern "C" {
#endif 

#ifndef __COMMON__
#define __COMMON__

/******************************************************************/

#define TRUE 1
#define FALSE 0
#define ERROR -1

#define STDIN_FILENAME "stdin"
#define STDOUT_FILENAME "stdout"
#define STDERR_FILENAME "stderr"


/******************************************************************/

#include <ctype.h>
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

/******************************************************************/

#include "data_structures.h"
#include "defines.h"
#include "error_handlers.h"
#ifndef __IMMINTRIN_H
#define __IMMINTRIN_H

#ifdef __MMX__
#include <mmintrin.h>
#endif

#ifdef __SSE__
#include <xmmintrin.h>
#endif

#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef __SSE3__
#include <pmmintrin.h>
#endif

#ifdef __SSSE3__
#include <tmmintrin.h>
#endif

#if defined (__SSE4_2__) || defined (__SSE4_1__)
#include <smmintrin.h>
#endif

#if defined (__AES__) || defined (__PCLMUL__)
#include <wmmintrin.h>
#endif

#ifdef __AVX__
#include <immintrin.h>
#endif

#endif /* __IMMINTRIN_H */

/******************************************************************/

#endif /* __COMMON__ */
#ifdef __cplusplus
}
#endif 
