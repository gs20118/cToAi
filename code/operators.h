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
    class Add : public Operator<T> {
    public:
        std::vector<Tensor<T>> backward(const Tensor<T>& dLdx){
            if(!Operator<T>::inputs[0].sameShape(dLdx)) throw std::domain_error("dLdx isn't same shape");
            return std::vector<Tensor<T>> {dLdx.clone(), dLdx.clone()};
        }
        Tensor<T> forward(const std::vector<Tensor<T>> &ten){
            if(!ten[0].sameShape(ten[1])) throw std::domain_error("two tensor isn't same shape");
            Tensor<T> o = ten[0].zero();
            o.foreach([&ten](Tensor<T> &this_, std::vector<int> &v){
                this_.item(v) = ten[0].item(v) + ten[1].item(v);
            });
            if(cai::isGrad && (ten[0].requires_grad || ten[1].requires_grad)){
                o.set_grad();
                o.grad_func = Operator<T>::shared_from_this();
            }
            return o;
        }
    };
    template<typename T>
    class Sub : public Operator<T>{
        std::vector<Tensor<T>> backward(const  Tensor<T>& dLdx){
            if(!Operator<T>::inputs[0].sameShape(dLdx)) throw std::domain_error("dLdx isn't same shape");
            Tensor<T> second = dLdx.clone();
            second.foreach([](Tensor<T> &this_, std::vector<int> &v){
                this_.item(v) = -this_.item(v);
            });
            return std::vector<Tensor<T>> {dLdx, second};
        }
        Tensor<T> forward(const  std::vector<Tensor<T>> &ten){
            if(!ten[0].sameShape(ten[1])) throw std::domain_error("two tensor shape isn't same");
            Tensor<T> o = ten[0].zero();
            o.foreach([&ten](Tensor<T> &this_, std::vector<int> &v){
                this_.item(v) = ten[0].item(v) - ten[1].item(v);
            });
            if(cai::isGrad && (ten[0].requires_grad || ten[1].requires_grad)){
                o.set_grad();
                o.grad_func = Operator<T>::shared_from_this();
            }
            return o;
        }
    };
    template<typename T>
    class Mult : public Operator<T>{
        std::vector<Tensor<T>> backward(const  Tensor<T>& dLdx){
            if(!Operator<T>::inputs[0].sameShape(dLdx)) throw std::domain_error("dLdx isn't same shape");
            return std::vector<Tensor<T>> {Operator<T>::inputs[1].detach(), Operator<T>::inputs[0].detach()};
        }
        Tensor<T> forward(const std::vector<Tensor<T>> &ten){
            if(!ten[0].sameShape(ten[1])) throw std::domain_error("two tensor shape isn't same");
            Tensor<T> o = ten[0].zero();
            o.foreach([&ten](Tensor<T> &this_, std::vector<int> &v){
                this_.item(v) = ten[0].item(v) * ten[1].item(v);
            });
            if(cai::isGrad && (ten[0].requires_grad || ten[1].requires_grad)){
                o.set_grad();
                o.grad_func = Operator<T>::shared_from_this();
            }
            return o;
        }
    };
    template<typename T>
    class Div : public Operator<T>{
        std::vector<Tensor<T>> backward(const  Tensor<T>& dLdx){
            Tensor<T> first = dLdx.zero(), second = dLdx.zero();
            std::vector<Tensor<T>> &vec = Operator<T>::inputs;
            first.foreach([&vec, &dLdx](Tensor<T> &this_, std::vector<int> &v){
                this_.item(v) = 1/vec[1].item(v);
            });
            second.foreach([&vec, &dLdx](Tensor<T> &this_, std::vector<int> &v){
                this_.item(v) = -vec[0].item(v)/(vec[1].item(v) * vec[1].item(v));
            });
            return std::vector<Tensor<T>> {first, second};
        }
        Tensor<T> forward(const  std::vector<Tensor<T>> &ten){
            if(!ten[0].sameShape(ten[1])) throw std::domain_error("two tensor shape isn't same");
            Tensor<T> o = ten[0].zero();
            o.foreach([&ten](Tensor<T> &this_, std::vector<int> &v){
                if(ten[1].item(v)==0) throw std::logic_error("can't divide by 0");
                this_.item(v) = ten[0].item(v) / ten[1].item(v);
            });
            if(cai::isGrad && (ten[0].requires_grad || ten[1].requires_grad)){
                o.set_grad();
                o.grad_func = Operator<T>::shared_from_this();
            }
            return o;
        }
    };

    template<typename T>
    class Square : public Operator<T>{
        std::vector<Tensor<T>> backward(const  Tensor<T>& dLdx){
            if(!Operator<T>::inputs[0].sameShape(dLdx)) throw std::domain_error("dLdx isn't same shape");
            Tensor<T> first = dLdx.zero();
            std::vector<Tensor<T>> &vec = Operator<T>::inputs;
            first.foreach([&vec, &dLdx](Tensor<T> &this_, std::vector<int> &v){
                this_.item(v) = 2 * vec[0].item(v) * dLdx.item(v);
            });
            return std::vector<Tensor<T>> {first};
        }
        Tensor<T> forward(const  std::vector<Tensor<T>> &ten){
            Tensor<T> o = ten[0].clone();
            o.foreach([](Tensor<T> &this_, std::vector<int> &v){
                T temp =  this_.item(v);
                this_.item(v) = temp*temp;
            });
            if(cai::isGrad && (ten[0].requires_grad)){
                o.set_grad();
                o.grad_func = Operator<T>::shared_from_this();
            }
            return o;
        }
    };
    template<typename T>
    class Exp : public Operator<T>{
        std::vector<Tensor<T>> backward(const  Tensor<T>& dLdx){
            if(!Operator<T>::inputs[0].sameShape(dLdx)) throw std::domain_error("dLdx isn't same shape");
            Tensor<T> first = dLdx.zero();
            std::vector<Tensor<T>> &ten = Operator<T>::inputs;
            first.foreach([&ten, &dLdx](Tensor<T> &this_, std::vector<int> &v){
                this_.item(v) = exp(ten[0].item(v)) * dLdx.item(v);
            });
            return std::vector<Tensor<T>> {first};
        }
        Tensor<T> forward(const  std::vector<Tensor<T>> &ten){
            Tensor<T> o = ten[0].clone();
            o.foreach([](Tensor<T> &this_, std::vector<int> &v){
                this_.item(v) = exp(this_.item(v));
            });
            if(cai::isGrad && (ten[0].requires_grad)){
                o.set_grad();
                o.grad_func = Operator<T>::shared_from_this();
            }
            return o;
        }
    };
    template<typename T>
    class Log : public Operator<T>{
        std::vector<Tensor<T>> backward(const  Tensor<T>& dLdx){
            if(!Operator<T>::inputs[0].sameShape(dLdx)) throw std::domain_error("dLdx isn't same shape");
            Tensor<T> first = dLdx.zero();
            std::vector<Tensor<T>> &ten = Operator<T>::inputs;
            first.foreach([&ten, &dLdx](Tensor<T> &this_, std::vector<int> &v){
                if(ten[0].item(v)==0) throw std::logic_error("can't divide by 0");
                this_.item(v) = (1/ten[0].item(v)) * dLdx.item(v);
            });
            return std::vector<Tensor<T>> {first};
        }
        Tensor<T> forward(const  std::vector<Tensor<T>> &ten){
            Tensor<T> o = ten[0].clone();

            o.foreach([](Tensor<T> &this_, std::vector<int> &v){
                this_.item(v) = log(this_.item(v));
            });
            if(cai::isGrad && (ten[0].requires_grad)){
                o.set_grad();
                o.grad_func = Operator<T>::shared_from_this();
            }
            return o;
        }
    };
    template<typename T>
    class Tanh : public Operator<T>{
        std::vector<Tensor<T>> backward(const  Tensor<T>& dLdx){
            if(!Operator<T>::inputs[0].sameShape(dLdx)) throw std::domain_error("dLdx isn't same shape");
            Tensor<T> first = dLdx.zero();
            std::vector<Tensor<T>> &ten = Operator<T>::inputs;
            first.foreach([&ten, &dLdx](Tensor<T> &this_, std::vector<int> &v){
                double temp = cosh(this_.item(v));
                this_.item(v) = (1/(temp*temp)) * dLdx.item(v);
            });
            return std::vector<Tensor<T>> {first};
        }
        Tensor<T> forward(const  std::vector<Tensor<T>> &ten){
            Tensor<T> o = ten[0].clone();
            o.foreach([](Tensor<T> &this_, std::vector<int> &v){
                this_.item(v) = tanh(this_.item(v));
            });
            if(cai::isGrad && (ten[0].requires_grad)){
                o.set_grad();
                o.grad_func = Operator<T>::shared_from_this();
            }
            return o;
        }
    };
    template<typename T>
    class ReLu : public Operator<T>{
        std::vector<Tensor<T>> backward(const  Tensor<T>& dLdx){
            if(!Operator<T>::inputs[0].sameShape(dLdx)) throw std::domain_error("dLdx isn't same shape");
            Tensor<T> first = dLdx.zero();
            std::vector<Tensor<T>> &ten = Operator<T>::inputs;
            first.foreach([&ten, &dLdx](Tensor<T> &this_, std::vector<int> &v){
                this_.item(v) = (ten[0].item(v) > 0 ? dLdx.item(v):0);
            });
            return std::vector<Tensor<T>> {first};
        }
        Tensor<T> forward(const  std::vector<Tensor<T>> &ten){
            Tensor<T> o = ten[0].clone();
            o.foreach([](Tensor<T> &this_, std::vector<int> &v){
                T& temp = this_.item(v);
                if(temp < 0) temp = 0;
            });
            if(cai::isGrad && (ten[0].requires_grad)){
                o.set_grad();
                o.grad_func = Operator<T>::shared_from_this();
            }
            return o;
        }
    };

    template<typename T>
    class Sum : public Operator<T>{
        std::vector<Tensor<T>> backward(const Tensor<T>& dLdx){
            std::vector<Tensor<T>> &vec = Operator<T>::inputs;
            Tensor<T> first = vec[0].zero();
            first.set(dLdx.item());
            return std::vector<Tensor<T>> {first};
        }
        Tensor<T> forward(const std::vector<Tensor<T>> &ten){
            Tensor<T> o = Tensor<T>(0);
            int num = ten[0].get_size();
            auto vec = ten[0].initi();
            while(num--){
                ten[0].nexti(vec);
                o.set(o.item() + ten[0].item(vec));
            }
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
            std::vector<Tensor<T>> &vec = Operator<T>::inputs;
            Tensor<T> first  = vec[0].zero();
            int num = vec[0].get_size();
            first.set(dLdx.item() / num);
            return std::vector<Tensor<T>> {first};
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
    class Cross : public Operator<T> {
    public:
        std::vector<Tensor<T>> backward(const Tensor<T>& dLdx){
            setGrad(false);
                Tensor<T> first = dLdx.cross(Operator<T>::inputs[1].trans());
                Tensor<T> second = Operator<T>::inputs[0].trans().cross(dLdx);
            setGrad(true);
            return std::vector<Tensor<T>> {first, second};
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
        std::shared_ptr<Operator<T>> op = std::shared_ptr<Operator<T>>(new Add<T>(), [](Operator<T>* a){delete a;});
        return (*op)(a, b);
    }
    template<typename T>
    Tensor<T> cross(const Tensor<T> &a, const Tensor<T> &b)  {
        std::shared_ptr<Operator<T>> op = std::shared_ptr<Operator<T>>(new Add<T>(), [](Operator<T>* a){delete a;});
        return (*op)(a, b);
    }
    template<typename T>
    Tensor<T> sub(const Tensor<T> &a, const Tensor<T> &b){
        std::shared_ptr<Operator<T>> op = std::shared_ptr<Operator<T>>(new Sub<T>(), [](Operator<T>* a){delete a;});
        return (*op)(a, b);
    }
    template<typename T>
    Tensor<T> mult(const Tensor<T> &a, const Tensor<T> &b){
        std::shared_ptr<Operator<T>> op = std::shared_ptr<Operator<T>>(new Mult<T>(), [](Operator<T>* a){delete a;});
        return (*op)(a, b);
    }
    template<typename T>
    Tensor<T> div(const Tensor<T> &a, const Tensor<T> &b){
        std::shared_ptr<Operator<T>> op = std::shared_ptr<Operator<T>>(new Div<T>(), [](Operator<T>* a){delete a;});
        return (*op)(a, b);
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
