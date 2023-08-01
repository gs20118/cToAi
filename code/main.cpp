#include "cai.h"
#include <ctime>

using namespace cai;
class SimpleNet : public Network{
public:
    void init(){
        set("linear1", new Linear(20, 20));
    }

    Tensor<double> forward(Tensor<double> &a){
        auto & l = Network::layers;
        a = l["linear1"]->forward(a);
        return a;
    }
};

void main_(){

    Tensor<double> x_t = rand(20, 100);
    Tensor<double> y_t = randn(20, 100);

    SimpleNet simplenet = SimpleNet();
    SimpleOptim<double> optimizer = SimpleOptim<double>(0.02);
    simplenet.init();
    optimizer.set(simplenet.parameters());
    optimizer.init();

    for(int Iter = 0; Iter < 1000; Iter++){
        Tensor<double> y_r = simplenet(x_t);
        Tensor<double> Loss = (y_r - y_t).square().sum();
        Loss.backward();
        optimizer.optim();
        simplenet.zero_grad();
        if(Iter == 999){
            std::cout << Loss << std::endl;
        }
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
