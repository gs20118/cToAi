//
// Created by choijunghwan on 2023-07-29.
//

#ifndef CLIONPROJECT_OPERATORS_H
#define CLIONPROJECT_OPERATORS_H

#include "tensor.h"
#include "operator.h"
#include <cmath>
namespace cai{
    template<typename T>
    class Add : public TwoToOneOperator<T> {
    public:
        T forward_func(T a, T b){return a + b;};
        T backward0_func(T a, T b, T L){return L;};
        T backward1_func(T a, T b, T L){return L;};
    };
    template<typename T>
    class Sub : public Operator<T>{
        T forward_func(T a, T b){return a - b;};
        T backward0_func(T a, T b, T L){return L;};
        T backward1_func(T a, T b, T L){return -L;};
    };
    template<typename T>
    class Mult : public Operator<T>{
        T forward_func(T a, T b){return a * b;};
        T backward0_func(T a, T b, T L){return b * L;};
        T backward1_func(T a, T b, T L){return a * L;};
    };
    template<typename T>
    class Div : public Operator<T>{
        T forward_func(T a, T b){
            if(a<=0) throw std::logic_error("can't divide by 0");
            return a / b;};
        T backward0_func(T a, T b, T L){return L/b;};
        T backward1_func(T a, T b, T L){return -a * L/ pow(b, 2);};
    };

    template<typename T>
    class Square : public OneToOneOperator<T>{
    public:
        T forward_func(T a){return a;};
        T backward_func(T a, T L){return 2*a*L;}
    };
    template<typename T>
    class Exp : public OneToOneOperator<T>{
        T forward_func(T a){return exp(a);};
        T backward_func(T a, T L){return exp(a) * L;}
    };
    template<typename T>
    class Log : public OneToOneOperator<T>{
        T forward_func(T a){
            if(a<=0) throw std::logic_error("o can't submit in log function");
            return log(a);};
        T backward_func(T a, T L){
            return  L/a;
        }
    };
    template<typename T>
    class Tanh : public OneToOneOperator<T>{
        T forward_func(T a){return tanh(a);};
        T backward_func(T a, T L){return L / pow(cosh(a), 2);}
    };
    template<typename T>
    class ReLu : public OneToOneOperator<T>{
        T forward_func(T a){return a > 0 ? a:0;};
        T backward_func(T a, T L){return a > 0 ? L : 0;}
    };

    template<typename T>
    class Sum : public Operator<T>{
        std::vector<Tensor<T>> backward(const Tensor<T>& dLdx){
            std::vector<Tensor<T>> &ten = Operator<T>::inputs;
            Tensor<T> gten0 = ten[0].zero();
            gten0.set(dLdx.item());
            return std::vector<Tensor<T>> {gten0};
        }
        Tensor<T> forward(const std::vector<Tensor<T>> &ten){
            Tensor<T> o = Tensor<T>(0);
            int num = ten[0].get_size();
            auto pos = ten[0].initi();
            while(num--){
                ten[0].nexti(pos);
                o.set(o.item() + ten[0].item(pos));
            }
            if(cai::isGrad && (ten[0].requires_grad)){
                o.set_grad();
                o.grad_func = Operator<T>::shared_from_this();
            }
            return o;
        }
    };

    template<typename T>
    class Max : public Operator<T>{
        std::vector<Tensor<T>> backward(const Tensor<T>& dLdx){
            std::vector<Tensor<T>> &ten = Operator<T>::inputs;
            Tensor<T> &oten = Operator<T>::output;
            Tensor<T> gten0 = ten[0].zero();
            gten0.foreach([&ten, &oten, &dLdx](Tensor<T> &this_, std::vector<int> &pos){
                this_.item(pos) = (oten.item() == ten[0].item(pos) ? dLdx.item() : 0);
            });
            return std::vector<Tensor<T>> {gten0};
        }
        Tensor<T> forward(const std::vector<Tensor<T>> &ten){
            Tensor<T> o = Tensor<T>(0);
            int num = ten[0].get_size()-1;
            auto pos = ten[0].initi();
            ten[0].nexti(pos);
            T val = ten[0].item(pos);
            while(num--){
                ten[0].nexti(pos);
                val = (std::max(val ,ten[0].item(pos)));
            }
            o.set(val);
            if(cai::isGrad && (ten[0].requires_grad)){
                o.set_grad();
                o.grad_func = Operator<T>::shared_from_this();
            }
            return o;
        }
    };
    template<typename T>
    class Min : public Operator<T>{
        std::vector<Tensor<T>> backward(const Tensor<T>& dLdx){
            std::vector<Tensor<T>> &ten = Operator<T>::inputs;
            Tensor<T> &oten = Operator<T>::output;
            Tensor<T> gten0 = ten[0].zero();
            gten0.foreach([&ten, &oten, &dLdx](Tensor<T> &this_, std::vector<int> &pos){
                this_.item(pos) = (oten.item() == ten[0].item(pos) ? dLdx.item() : 0);
            });
            return std::vector<Tensor<T>> {gten0};
        }
        Tensor<T> forward(const std::vector<Tensor<T>> &ten){
            Tensor<T> o = Tensor<T>(0);
            int num = ten[0].get_size()-1;
            auto pos = ten[0].initi();
            ten[0].nexti(pos);
            T val = ten[0].item(pos);
            while(num--){
                ten[0].nexti(pos);
                val = (std::min(val ,ten[0].item(pos)));
            }
            o.set(val);
            if(cai::isGrad && (ten[0].requires_grad)){
                o.set_grad();
                o.grad_func = Operator<T>::shared_from_this();
            }
            return o;
        }
    };

    template<typename T>
    class Mean : public Operator<T>{
        std::vector<Tensor<T>> backward(const Tensor<T>& dLdx){
            std::vector<Tensor<T>> &ten = Operator<T>::inputs;
            Tensor<T> gten0  = ten[0].zero();
            int num = ten[0].get_size();
            gten0.set(dLdx.item() / num);
            return std::vector<Tensor<T>> {gten0};
        }
        Tensor<T> forward(const  std::vector<Tensor<T>> &ten){
            Tensor<T> o = Tensor<T>(0);
            int num = ten[0].get_size();
            std::vector<int> vec = ten[0].initi();
            for(int i=0; i<num; i++){
                ten[0].nexti(vec);
                o.set(o.item() + ten[0].item(vec));
            }
            o.set(o.item()/num);
            if(cai::isGrad && (ten[0].requires_grad)){
                o.set_grad();
                o.grad_func = Operator<T>::shared_from_this();
            }
            return o;
        }
    };

    template<typename T>
    class Normal : public Operator<T>{
        std::vector<Tensor<T>> backward(const Tensor<T>& dLdx){
            if(!Operator<T>::inputs[0].sameShape(dLdx)) throw std::domain_error("dLdx isn't same shape");
            std::vector<Tensor<T>> &ten = Operator<T>::inputs;
            Tensor<T> gten0 = dLdx.zero();
            setGrad(false);
                Tensor<T> sum = ten[0].sum();
            setGrad(true);
            gten0.foreach([&ten, &dLdx, &sum, this](Tensor<T> &this_, std::vector<int> &pos){
                this_.item(pos) = dLdx.item(pos) * ((1/sum.item() - ten[0].item(pos) / pow(sum.item(), 2))) ;
            });
            return std::vector<Tensor<T>> {gten0};
        }
        Tensor<T> forward(const  std::vector<Tensor<T>> &ten){
            Tensor<T> o = ten[0].clone();
            setGrad(false);
                Tensor<T> sum = o.sum();
            setGrad(true);
            if(sum.item() == 0) throw std::logic_error("Can't devide by 0");
            o.foreach([&ten, &sum](Tensor<T> &this_, std::vector<int> &pos){
                this_.item(pos) = ten[0].item(pos) / sum.item();
            });
            if(cai::isGrad && (ten[0].requires_grad)){
                o.set_grad();
                o.grad_func = Operator<T>::shared_from_this();
            }
            return o;
        }
    };

    template<typename T>
    class Cross : public Operator<T> {
    public:
        std::vector<Tensor<T>> backward(const Tensor<T>& dLdx){
            setGrad(false);
                Tensor<T> gten0 = dLdx.cross(Operator<T>::inputs[1].trans());
                Tensor<T> gten1 = Operator<T>::inputs[0].trans().cross(dLdx);
            setGrad(true);
            return std::vector<Tensor<T>> {gten0, gten1};
        }
        Tensor<T> forward(const std::vector<Tensor<T>> &ten){
            if(ten[0].dim != ten[1].dim) throw std::domain_error("two tensor isn't same dimension");
            if(ten[0].dim <= 1) throw std::domain_error("if dimension under 1 isn't defined");
            if(ten[0].shape[ten[0].dim-1] != ten[1].shape[ten[1].dim-2]) throw std::domain_error("two tensor don't define cross multipleation");
            int newDim = ten[0].dim;
            int* newShape = func::newArray_(newDim, ten[0].shape);
            newShape[newDim-1] = ten[1].shape[newDim-1];
            Tensor<T> o = Tensor<T>(newDim, newShape, 0);

            o.foreach([&ten](Tensor<T> &this_, std::vector<int> &v){
                std::vector<int> v1 = v, v2 = v;
                for(int i=0; i<ten[0].shape[ten[0].dim-1]; i++){
                    v1[ten[0].dim-1] = i;
                    v2[ten[1].dim-2] = i;
                    this_.item(v) += ten[0].item(v1) * ten[1].item(v2);
                }
            });

            if(cai::isGrad && (ten[0].requires_grad || ten[1].requires_grad)){
                o.set_grad();
                o.grad_func = Operator<T>::shared_from_this();
            }
            return o;
        }
    };

    template<typename T>
    Tensor<T> add(const Tensor<T> &a, const Tensor<T> &b)  {
        return a+b;
    }
    template<typename T>
    Tensor<T> cross(const Tensor<T> &a, const Tensor<T> &b)  {
        return a.cross(b);
    }
    template<typename T>
    Tensor<T> sub(const Tensor<T> &a, const Tensor<T> &b){
        return a-b;
    }
    template<typename T>
    Tensor<T> mult(const Tensor<T> &a, const Tensor<T> &b){
        return a*b;
    }
    template<typename T>
    Tensor<T> div(const Tensor<T> &a, const Tensor<T> &b){
        return a/b;
    }

    template<typename T>
    Tensor<T> square(const Tensor<T> &a){
        std::shared_ptr<Operator<T>> op = std::shared_ptr<Operator<T>>(new Square<T>(), [](Operator<T>* a){delete a;});
        return (*op)(a);
    }
    template<typename T>
    Tensor<T> exp(const Tensor<T> &a){
        std::shared_ptr<Operator<T>> op = std::shared_ptr<Operator<T>>(new Exp<T>(), [](Operator<T>* a){delete a;});
        return (*op)(a);
    }
    template<typename T>
    Tensor<T> log(const Tensor<T> &a){
        std::shared_ptr<Operator<T>> op = std::shared_ptr<Operator<T>>(new Log<T>(), [](Operator<T>* a){delete a;});
        return (*op)(a);
    }
    template<typename T>
    Tensor<T> tanh(const Tensor<T> &a){
        std::shared_ptr<Operator<T>> op = std::shared_ptr<Operator<T>>(new Tanh<T>(), [](Operator<T>* a){delete a;});
        return (*op)(a);
    }
    template<typename T>
    Tensor<T> relu(const Tensor<T> &a){
        std::shared_ptr<Operator<T>> op = std::shared_ptr<Operator<T>>(new ReLu<T>(), [](Operator<T>* a){delete a;});
        return (*op)(a);
    }
    template<typename T>
    Tensor<T> normal(const Tensor<T> &a){
        std::shared_ptr<Operator<T>> op = std::shared_ptr<Operator<T>>(new Normal<T>(), [](Operator<T>* a){delete a;});
        return (*op)(a);
    }
    template<typename T>
    Tensor<T> soft_max(const Tensor<T> &a){
        std::shared_ptr<Operator<T>> op1 = std::shared_ptr<Operator<T>>(new Exp<T>(), [](Operator<T>* a){delete a;});
        std::shared_ptr<Operator<T>> op2 = std::shared_ptr<Operator<T>>(new Normal<T>(), [](Operator<T>* a){delete a;});
        return (*op2)((*op1)(a));
    }

    template<typename T>
    Tensor<T> max(const Tensor<T> &a){
        std::shared_ptr<Operator<T>> op = std::shared_ptr<Operator<T>>(new Max<T>(), [](Operator<T>* a){delete a;});
        return (*op)(a);
    }

    template<typename T>
    Tensor<T> min(const Tensor<T> &a){
        std::shared_ptr<Operator<T>> op = std::shared_ptr<Operator<T>>(new Max<T>(), [](Operator<T>* a){delete a;});
        return (*op)(a);
    }

    template<typename T>
    Tensor<T> sum(const Tensor<T> &a){
        std::shared_ptr<Operator<T>> op = std::shared_ptr<Operator<T>>(new Sum<T>(), [](Operator<T>* a){delete a;});
        return (*op)(a);
    }
    template<typename T>
    Tensor<T> mean(const Tensor<T> &a){
        std::shared_ptr<Operator<T>> op = std::shared_ptr<Operator<T>>(new Mean<T>(), [](Operator<T>* a){delete a;});
        return (*op)(a);
    }
}

#endif //CLIONPROJECT_OPERATORS_H
