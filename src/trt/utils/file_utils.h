#ifndef __FILE_UTILS_H__
#define __FILE_UTILS_H__
#include <string>

namespace FileUtils
{
    bool saveFile(const std::string& file, const void* data, size_t len);

} // namespace FileUtils


#endif // __FILE_UTILS_H__