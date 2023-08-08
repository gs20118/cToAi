#ifndef CLIONPROJECT_CAILIB_H
#define CLIONPROJECT_CAILIB_H
#include <initializer_list>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <map>

namespace cai
{
    template<typename T>
    class Tensor;
    template<typename T>
    class Operator;
    class Range;

    class Layer;
    class Network;
    class Optimizer;

    bool isGrad = true;
    void setGrad(bool grad){
        isGrad = grad;
    }
    
}

#endif //CLIONPROJECT_CAILIB_H
