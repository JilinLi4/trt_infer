#include "trt_builder.h"
#include "spd_logger.h"
#include "trt_logger.h"
#include "trt/utils/trt_uitls.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

#include <cuda_runtime_api.h>
#include <NvOnnxParser.h>

using namespace std;
namespace TRT
{

	static string join_dims(const vector<int>& dims){
		stringstream output;
		char buf[64];
		const char* fmts[] = {"%d", " x %d"};
		for(int i = 0; i < dims.size(); ++i){
			snprintf(buf, sizeof(buf), fmts[i != 0], dims[i]);
			output << buf;
		}
		return output.str();
	}

const char* model_string(ModelType type) {
    switch (type) {
    case ModelType::FP32:
        return "FP32";
    case ModelType::FP16:
        return "FP16";
    default:
        return "UnknowTRTMode";
    }
}

bool compile(ModelType model_type,
            unsigned int maxBatchSize,
            const std::string& onnxPath,
            const std::string& savePath,
            const size_t maxWorkspaceSize) {

    LOG_INFO("Compile {} {}.", model_string(model_type), onnxPath);
    static TRTLogger trtLogger;
    auto builder = make_nvshared(nvinfer1::createInferBuilder(trtLogger));
    if (builder == nullptr) {
        LOG_INFOE("Can not create builder.");
        return false;
    }

    auto config = make_nvshared(builder->createBuilderConfig());
    if (model_type == ModelType::FP16) {
        if (!builder->platformHasFastFp16()) {
            LOG_INFOW("Platform not have fast fp16 support");
        }
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    std::shared_ptr<nvinfer1::INetworkDefinition> network;
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    network = make_nvshared(builder->createNetworkV2(explicitBatch));

    std::shared_ptr<nvonnxparser::IParser> onnxParser = make_nvshared(nvonnxparser::createParser(*network, trtLogger));
    if (onnxParser == nullptr) {
        LOG_INFOE("Can not create parser.");
        return false;
    }

    if (!onnxParser->parseFromFile(onnxPath.c_str(), 1)) {
        LOG_INFOE("Can not parse OnnX file:{}", onnxPath.c_str());
        return false;
    }

    auto inputTensor = network->getInput(0);
    auto inputDims = inputTensor->getDimensions();

    LOG_INFO("Input shape is {}", join_dims(vector<int>(inputDims.d, inputDims.d + inputDims.nbDims)));
    LOG_INFO("Set max batch size = {}", maxBatchSize);
    LOG_INFO("Set max workspace size = {} MB", maxWorkspaceSize / 1024.0f / 1024.0f);




}


} // namespace TRT
