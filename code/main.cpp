#include "cai.h"
#include <ctime>

using namespace cai;
class SimpleNet : public Network{
public:
    void init() override{
        set("linear1", new Linear(20, 20));
    }

    Tensor<double> forward(Tensor<double> &a) override{
        auto & l = Network::layers;
        auto ret  = l["linear1"]->forward(a);
        return ret;
    }
};

void main_(){

    Tensor<double> x_t = rand(20, 1);
    Tensor<double> y_t = randn(20, 1);
    Tensor<double> test;

    SimpleNet simplenet = SimpleNet();
    SimpleOptim<double> optimizer = SimpleOptim<double>(0.01);
    simplenet.init();
    optimizer.set(simplenet.parameters());
    optimizer.init();

    for(int Iter = 0; Iter < 1000; Iter++){
        Tensor<double> y_r = simplenet(x_t);
        Tensor<double> Z = (y_r - y_t).square().mean();
        Z.backward();
        optimizer.optim();
        simplenet.zero_grad();
        std::cout << Iter << ": " << Z << std::endl;
    }

}

int main() {
    clock_t start, finish;
    double duration;
    start = clock();
    main_();
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    std::cout << duration << "s" << std::endl;
    return 0;
}
