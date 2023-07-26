#include "cai.h"
#include <ctime>

int main() {
    clock_t start, finish;
    double duration;
    start = clock();

    cai::Tensor<int> X = cai::arange<int>(20).reshape(4, 5);
    cai::Tensor<int> Y = cai::randint(0, 100, 4, 5);
    std::cout << X + Y << std::endl;

    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    std::cout << duration << "s" << std::endl;
    return 0;
}
