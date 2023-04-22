#ifndef __TRT_UITLS_H__
#define __TRT_UITLS_H__
#include <memory>

namespace TRT
{

template<typename _T>
std::shared_ptr<_T> make_nvshared(_T* ptr) {
    return std::shared_ptr<_T>(ptr, [](_T* p){p->destroy();});
}

} // namespace TRT
#endif // __TRT_UITLS_H__