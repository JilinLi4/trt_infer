#include "test_head.h"
#include "spd_logger.h"
#include "common/cuda_tools.h"

TEST_CASE("cuda_tool") {
    std::string rst = CUDATools::device_description(0);
    LOG_INFO("{}",rst.c_str());
}