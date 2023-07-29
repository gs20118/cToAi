#ifndef CLIONPROJECT_OPERATOR_H
#define CLIONPROJECT_OPERATOR_H
#include "cailib.h"
#include "variable.h"

/*
namespace cai::Operation
{
    template<typename To, typename... Froms>
    class Operator{
        std::tuple<Froms...> inputs;
        To output;

        To operator()(Froms... v){
            inputs = {v...};
            output = forward(v...);
            return output;
        }
        virtual To forward(Froms... v);
        virtual std::tuple<Froms...> backward(To dLdx);
    };

    template<typename T>
    class Add : Operator<Variable<T>, Variable<T>, Variable<T>>
    {
        Variable<T> forward(Variable<T> a, Variable<T> b){
            Tensor<T> o = a.body.copy();
            o.foreach([&a, &b](Tensor<T>& this_, std::vector<int> pos){
                this_.item(pos) = a.body.item(pos) + b.body.item(pos);
            });
            return Variable<T>(o);
        }
        std::tuple<Variable<T>, Variable<T>> backward(variable<T> dLdx){
            return std::tuple<Variable<T>, Variable<T>>{dLdx, dLdx};
        }
    };
}
*/
#endif //CLIONPROJECT_OPERATOR_H
