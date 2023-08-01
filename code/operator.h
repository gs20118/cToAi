#ifndef CLIONPROJECT_OPERATOR_H
#define CLIONPROJECT_OPERATOR_H
#include "cailib.h"
#include <vector>
#include <memory>

namespace cai
{
    template<typename T>
    class Operator : public std::enable_shared_from_this<Operator<T>>{
    public:
        int gen = 0;
        std::vector<Tensor<T>> inputs;
        Tensor<T> output;
        template<typename... Inputs>
        Tensor<T> operator()(Inputs... inputs){
            this->inputs = std::vector<Tensor<T>>{inputs.copy()...};
            output = forward(this->inputs);
            for(auto &t :this->inputs){
                if(t.grad_func != nullptr){
                    gen = std::max(gen, t.grad_func.get()->gen + 1);
                }
            }
            return output;
        }
        virtual Tensor<T> forward(const std::vector<Tensor<T>> &ten){
            return ten[0];
        }
        virtual std::vector<Tensor<T>> backward(const Tensor<T> &dLdx){
            return std::vector<Tensor<T>>{dLdx};
        }
    };

    template<typename T>
    class OneToOneOperator : Operator<T>{
    public:
        std::vector<Tensor<T>> backward(const  Tensor<T>& dLdx){
            if(!Operator<T>::inputs[0].sameShape(dLdx)) throw std::domain_error("dLdx isn't same shape");
            Tensor<T> first = dLdx.zero();
            std::vector<Tensor<T>> &ten = Operator<T>::inputs;
            first.foreach([&ten, &dLdx, this](Tensor<T> &this_, std::vector<int> &v){
                this_.item(v) = backward_fun(ten[0].item(v), dLdx.item(v));
            });
            return std::vector<Tensor<T>> {first};
        }
        virtual T forward_fun(T a){return a;};
        virtual T backward_fun(T a, T b){return a;};
        Tensor<T> forward(const  std::vector<Tensor<T>> &ten){
            Tensor<T> o = ten[0].zero();
            o.foreach([&ten, this](Tensor<T> &this_, std::vector<int> &v){
                this_.item(v) = forward_fun(ten[0].item(v));
            });
            if(cai::isGrad && (ten[0].requires_grad)){
                o.set_grad();
                o.grad_func = Operator<T>::shared_from_this();
            }
            return o;
        }
    };

    template<typename T> class Add;
    template<typename T> class Sub;
    template<typename T> class Mult;
    template<typename T> class Div;
    template<typename T> class Square;
    template<typename T> class Exp;
    template<typename T> class Log;
    template<typename T> class Tanh;
    template<typename T> class Sum;
    template<typename T> class Mean;
    template<typename T> class Cross;

}

template<typename T>
std::ostream& operator<<(std::ostream& o, const cai::Operator<T>& a){
    o << a.output;
    return o;
}

template<typename T>
bool operator<(const std::shared_ptr<cai::Operator<T>>& a, const std::shared_ptr<cai::Operator<T>>& b){
    return a.get()->gen > b.get()->gen;
}

#endif //CLIONPROJECT_OPERATOR_H
