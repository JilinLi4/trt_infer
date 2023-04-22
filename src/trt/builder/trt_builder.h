#ifndef __TRT_BUILDER_H__
#define __TRT_BUILDER_H__

#include <vector>
#include <string>
#include "spd_logger.h"

namespace TRT
{
    /**
     * @brief Model Quantization Precision Type
     */
    enum class ModelType : int {
        FP32,
        FP16,
        INT8
    };

    const char* model_string(ModelType type);


    /**
     * @brief Compile onnx to tensorRT engine
     * mode:				FP32 | FP16
    * source:				onnx path
	 * saveto				tensorRT engine file save path
	 * maxWorkspaceSize		maxWorkspaceSize  1ul << 30 = 1GB ul == unsigned long
     */
    bool compile(
            ModelType model,
            unsigned int maxBatchSize,
            const std::string& onnxPath,
            const std::string& savePath,
            const size_t maxWorkspaceSize = 1ul << 30
        );


} // namespace TRT


#endif // __TRT_BUILDER_H__