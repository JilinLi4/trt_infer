#include <stdio.h>
#include <chrono>
#include <thread>

using namespace std;

void worker(int a, std::string& output) {
    printf("hello thread!\n");
    this_thread::sleep_for(chrono::milliseconds(1000));
    output = "work output";
    printf("worker done.\n");
}

class Infer {
public:
    Infer() {
        worker_thread_ = thread(&Infer::infer_worker, this);
    }
    ~Infer() {
        if(worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }

private:
    void infer_worker() {
        for (size_t i = 0; i < 100; i++)
        {
            printf("hello thread!\n");
            this_thread::sleep_for(chrono::milliseconds(2000));
        }

    }

private:
    thread worker_thread_;
};



int main() {

    Infer infer;
    std::string output;
    thread t(worker, 567, std::ref(output));

    if (t.joinable()) {
        t.join();
    }
    printf("output: %s\n", output);
    printf("main done.\n");
    return 0;
}
