#include "file_utils.h"
#include <fstream>
#include <iostream>

using namespace std;
namespace FileUtils
{
    bool saveFile(const string& file, const void* data, size_t len) {
        FILE* f = fopen(file.c_str(), "wb");
        if(!f) {
            return false;
        } 
        if(data && len > 0) {
            if (fwrite(data, 1, len, f) != len) {
                fclose(f);
                return false;
            }
        }
        fclose(f);
        return true;
    }
} // namespace FileUtils
