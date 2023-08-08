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
            this->inputs = std::vector<Tensor<T>>{inputs...};
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
    class OneToOneOperator : public Operator<T>{
    public:
        std::vector<Tensor<T>> backward(const  Tensor<T>& dLdx){
            if(!Operator<T>::inputs[0].sameShape(dLdx)) throw std::domain_error("dLdx isn't same shape");
            Tensor<T> gten0 = dLdx.zero();
            std::vector<Tensor<T>> &ten = Operator<T>::inputs;
            gten0.foreach([&ten, &dLdx, this](Tensor<T> &this_, std::vector<int> &pos){
                this_.item(pos) = backward_func(ten[0].item(pos), dLdx.item(pos));
            });
            return std::vector<Tensor<T>> {gten0};
        }
        virtual T forward_func(T a){return a;};
        virtual T backward_func(T a, T L){return a;};
        Tensor<T> forward(const  std::vector<Tensor<T>> &ten){
            Tensor<T> o = ten[0].zero();
            o.foreach([&ten, this](Tensor<T> &this_, std::vector<int> &pos){
                this_.item(pos) = forward_func(ten[0].item(pos));
            });
            if(cai::isGrad && (ten[0].requires_grad)){
                o.set_grad();
                o.grad_func = Operator<T>::shared_from_this();
            }
            return o;
        }
    };

    template<typename T>
    class TwoToOneOperator : public Operator<T>{
    public:
        virtual T forward_func(T a, T b){return a;};
        virtual T backward0_func(T a, T b, T L){return a;};
        virtual T backward1_func(T a, T b, T L){return b;};
        std::vector<Tensor<T>> backward(const  Tensor<T>& dLdx){
            Tensor<T> gten0 = dLdx.zero(), gten1 = dLdx.zero();
            std::vector<Tensor<T>> &ten = Operator<T>::inputs;
            gten0.foreach([&ten, &dLdx, this](Tensor<T> &this_, std::vector<int> &pos){
                this_.item(pos) = backward0_func(ten[0].item(pos), ten[1].item(pos), dLdx.item(pos));
            });
            gten1.foreach([&ten, &dLdx, this](Tensor<T> &this_, std::vector<int> &pos){
                this_.item(pos) = backward1_func(ten[0].item(pos), ten[1].item(pos), dLdx.item(pos));
            });
            return std::vector<Tensor<T>> {gten0, gten1};
        }
        Tensor<T> forward(const  std::vector<Tensor<T>> &ten){
            if(!ten[0].sameShape(ten[1])) throw std::domain_error("two tensor shape isn't same");
            Tensor<T> o = ten[0].zero();
            o.foreach([&ten, this](Tensor<T> &this_, std::vector<int> &v){
                this_.item(v) = forward_func(ten[0].item(v), ten[1].item(v));
            });
            if(cai::isGrad && (ten[0].requires_grad || ten[1].requires_grad)){
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
    template<typename T> class Normal;
    template<typename T> class Sum;
    template<typename T> class Mean;
    template<typename T> class Max;
    template<typename T> class Min;
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
