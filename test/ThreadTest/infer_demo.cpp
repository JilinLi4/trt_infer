#include "infer.hpp"
#include <stdio.h>
#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <future>



struct Job {
	std::shared_ptr<std::promise<std::string>> pro;
	std::string input;
};

class InferImpl : public InferInterface {
public:
	virtual ~InferImpl() {
		stop();
	}
	bool load_model(const std::string& file) {

		//  使得资源哪里分配哪里释放，哪里使用，这样能够使得程序能够足够简单
		std::promise<bool> pro;
		worker_thread_ = std::thread(&InferImpl::worker, this, file, std::ref(pro));
		return pro.get_future().get();
	}

	void stop() {
		if (is_running_) {
			is_running_ = false;
			// 退出worker线程的等待
			cv_.notify_one();
		}

		//  保证推理线程结束，防止成为孤儿线程
		if (this->worker_thread_.joinable()) {
			worker_thread_.join();
		}
	}
	
	virtual std::shared_future<std::string> commit(const std::string& input) override {
		Job job;
		job.pro.reset(new std::promise<std::string>);
		job.input = input;
		//std::this_thread::sleep_for(std::chrono::milliseconds(100));

		std::shared_future<std::string> fut = job.pro->get_future();
		{
			std::lock_guard<std::mutex> l(lock_);
			qjobs_.emplace(job);
		}
		// 被动通知，有任务发送给worker
		cv_.notify_one();
		return fut;
	}

	// 实际执行模型的部分
	//void worker(std::string& file, std::promise<bool>& pro) {
	void worker(std::string file, std::promise<bool>& pro) {

		//std::string file = "aaa";
		// worker 内实现，模型的加载、使用、释放
		std::string context_ = file;
		if (context_.empty()) {
			pro.set_value(false);
			return;
		}
		else {
			is_running_ = true;
			pro.set_value(true);
		}
		int max_batch_size = 5;
		std::vector<Job> jobs;
		int batch_ids = 0;
		while (is_running_) {
			//  在队列取任务并执行的过程
			{
				std::unique_lock<std::mutex> l(lock_);
				cv_.wait(l, [&]() {
					return !is_running_ || !qjobs_.empty();
				});

				if (!is_running_) break;
				while (jobs.size() < max_batch_size && !qjobs_.empty()) {
					jobs.emplace_back(std::move(qjobs_.front()));
					qjobs_.pop();
				}

				// batch process
				for (auto& job : jobs) {
					char buff[100];
					sprintf_s(buff, "%s ---processed[%d]", job.input.c_str(), batch_ids);
					job.pro->set_value(buff);
				}
				std::this_thread::sleep_for(std::chrono::milliseconds(1500));
				batch_ids++;
				jobs.clear();

			} // end unique_lock

			
		}
		printf("[%s] Infer worker done. \n", file.c_str());

	}

private:
	std::thread					worker_thread_;		// 消费者
	std::queue<Job>				qjobs_;
	std::mutex					lock_;
	std::condition_variable		cv_;
	std::atomic<bool>			is_running_{false};

};


std::shared_ptr<InferInterface> create_infer(const std::string& file) {
	std::shared_ptr<InferImpl> instance(new InferImpl());
	if (!instance->load_model(file)) {
		instance.reset();
	}
	return instance;
}