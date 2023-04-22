#ifndef __CUDA_TOOLS_H__
#define __CUDA_TOOLS_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

#define checkCudaRuntime(call) CUDATools::checkRuntime(call, #call, __LINE__, __FILE__)

namespace CUDATools
{

bool checkRuntime(cudaError_t e, const char* call, int iLine, const char *szFile);

std::string device_description(int device_id);

} //namespace CUDATools


#endif // __CUDA_TOOLS_H__