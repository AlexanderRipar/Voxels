#pragma once

#include "och_debug.h"

#define CHECK(call) if(auto err = (call)) OCH_ERRLOG(cudaGetErrorString(err));

#define CHECKED_TYPEMALLOC(ptr, cnt) (cudaMalloc((void**)&(ptr), (cnt) * sizeof(*(ptr))));

#define CHECKED_MALLOC(ptr, cnt) (cudaMalloc((void**)&(ptr), (cnt)));
