#include "cuda_tools.h"
#include "spd_logger.h"
#include <fmt/format.h>

namespace CUDATools
{

bool checkRuntime(cudaError_t e, const char* call, int line, const char *file){
    if (e != cudaSuccess) {
        LOG_INFOE("CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d", 
            call, 
            cudaGetErrorString(e), 
            cudaGetErrorName(e), 
            e, file, line
        );
        return false;
    }
    return true;
}

std::string device_description(int device_id)
{
    cudaDeviceProp prop;
    size_t free_mem, total_mem;

    checkCudaRuntime(cudaGetDevice(&device_id));
    checkCudaRuntime(cudaGetDeviceProperties(&prop, device_id));
    checkCudaRuntime(cudaMemGetInfo(&free_mem, &total_mem));
    return fmt::format(
            "[ID {}]<{}>[arch [{}.{}][GMEM {:.2f} GB / {:.2f} GB]",
            device_id, prop.name, prop.major, prop.minor, 
            free_mem / 1024.0f / 1024.0f / 1024.0f,
            total_mem / 1024.0f / 1024.0f / 1024.0f
    );
}

} // namespace CudaTools