#include <stdio.h>
#include <thread>
#include <mutex>
#include <string>
#include <queue>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <future>
#include <memory>

using namespace std;
struct Job {
    shared_ptr<promise<string>> pro;
    string input;
};

mutex lock_;
queue<Job> qjobs_;
condition_variable cv_;
const int limit_ = 10;
void video_capture() {
    int pic_id = 0;
    while(true) {
        Job job;
        {
            unique_lock lock(lock_);
            // cv_.wait() 当条件满足时 获得锁的占有权，继续执行
            // 当条件不满足时，释放锁，线程阻塞等待
            cv_.wait(lock, [&](){
                return qjobs_.size() < limit_;
            });
            char name[100];
            sprintf(name, "PIC_%d", pic_id++);
            printf("生产了一个新图片: %s qjob.size(): %d\n", name, qjobs_.size());
            job.pro.reset(new promise<string>());
            job.input = name;
            qjobs_.push(job);
        }
        auto result = job.pro->get_future().get();
        printf("JOB %s -> %s\n", job.input.c_str(), result);
        this_thread::sleep_for(chrono::milliseconds(2000));
    }
}

void infer_worker() {
    while (true)
    {
        if(!qjobs_.empty()) {
            {
                unique_lock lock(lock_);
                auto pjob = qjobs_.front();
                qjobs_.pop();
                printf("消费掉一个图片: %s\n", pjob.input);
                // 消费一个后通知 cv_ 并释放锁
                auto result = pjob.input + "--infer";
                pjob.pro->set_value(result);
                cv_.notify_one();
            }
            this_thread::sleep_for(chrono::milliseconds(4000));
        }
        // 强制当前线程交出时间片，防止一直占用cpu资源
        this_thread::yield();
    }
    
}

int main() {
    thread t1(video_capture);
    thread t2(infer_worker);
    if (t1.joinable()) {
        t1.join();
    }
    if (t2.joinable()) {
        t2.join();
    }
    printf("Done!");
    return 0;
}