#include <NvInfer.h>
#include "spd_logger.h"

namespace TRT
{

class TRTLogger : public nvinfer1::ILogger
{
public:
    virtual void log(Severity severity, const char *msg) noexcept override
    {

        if (severity == Severity::kINTERNAL_ERROR)
        {
            LOG_INFOE("NVInfer INTERNAL_ERROR: {}", msg);
            abort();
        }
        else if (severity == Severity::kERROR)
        {
            LOG_INFOE("NVInfer: {}", msg);
        }
        else if (severity == Severity::kWARNING)
        {
            LOG_INFOW("NVInfer: {}", msg);
        }
        else if (severity == Severity::kINFO)
        {
            LOG_INFO("NVInfer:{}", msg);
        }
        else
        {
            LOG_INFO("{}", msg);
        }
    }

    static TRTLogger trtLogger;
};

} // namespace TRT