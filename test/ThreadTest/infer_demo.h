#ifndef INFER_HPP
#define INFER_HPP

#include <memory>
#include <string>
#include <future>

/**
接口类 是一个纯虚函数
原则上只暴露调用者需要的函数，其他一概不暴露
比如说 load_model, 通过RAII做定义，因此load_model 属于不需要的范畴
内部如果启动线程等，start， stop,也不需要暴露，而是初始化的时候就自动启动，都是RAII的定义

*/
class InferInterface {
public:
	//virtual std::shared_future<std::string> forward(std::string pic) = 0;
	virtual std::shared_future<std::string> commit(const std::string& input) = 0;
};
std::shared_ptr<InferInterface> create_infer(const std::string& file);


#endif // !INFER_HPP
