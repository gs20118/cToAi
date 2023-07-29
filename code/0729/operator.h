//
// Created by 최정환 on 2023-07-29.
//

#ifndef NEWPROJECTS_OPERATOR_H
#define NEWPROJECTS_OPERATOR_H

#include "cailib.h"
#include <vector>

namespace cai
{
    template<typename T>
    class Operator{
        std::vector<Tensor<T>> inputs;
        Tensor<T> output;
        template<typename... Inputs>
        Tensor<T> operator()(Inputs... inputs){
            this->inputs = std::vector<Tensor<T>>{inputs...};
            output = forward(inputs...);
            return output;
        }
        template<typename... Inputs>
        Tensor<T> forward(Inputs... inputs);
        virtual std::vector<Tensor<T>> backward(Tensor<T>&);
    };

    template<typename T>
    class Add : Operator<T>{};

    template<typename T>
    class Sub;

    template<typename T>
    class Mult;
}

#endif //NEWPROJECTS_OPERATOR_H
