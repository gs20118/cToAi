#ifndef CAIPROJECT_VARIABLE_H
#define CAIPROJECT_VARIABLE_H
#include "cailib.h"
#include "tensor.h"
/*
namespace cai
{
    template<typename T>
    class Variable {
        Tensor<T> body, grad;
        bool requires_grad = false;

        Variable(std::initializer_list<T> val, std::initializer_list<T> sh){
            body = Tensor<T>(val, sh);
        }

        Variable(Tensor<T> &body){
            this->body = body;
        }

        Variable(const Variable &o){
            body = o.body;
            grad = o.grad;
        }

        void set_grad(){
            requires_grad = true;
            grad = Tensor<T>(body.dim, func::newArray(body.dim, body.shape), 0);
        }

        void print(){
            print(std::cout);
        }
        void print(std::ostream &o){
            body.print(o);
        }

    };
}

template<typename T>
std::ostream& operator<<(std::ostream &o, const cai::variable<T> v){
    v.print(o);
}
*/

#endif //CAIPROJECT_VARIABLE_H
