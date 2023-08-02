//
// Created by choijunghwan on 2023-08-01.
//

#ifndef CLIONPROJECT_LAYER_H
#define CLIONPROJECT_LAYER_H
#include "tensor.h"
#include "operators.h"
#include "cailib.h"
#include <vector>

namespace cai{
    class Layer{
    public:
        virtual Tensor<double> forward(const Tensor<double> &a){
            return a;
        }
        Tensor<double> operator()(const Tensor<double> &a){
            return forward(a);
        }
        virtual void zero_grad(){}
        virtual std::vector<std::tuple<double *, double *, int>> parameters(){
            return std::vector<std::tuple<double *, double *, int>>();
        }
        virtual void print_grad(){}
    };

    class Linear:public Layer{
    public:
        Tensor<double> W, b;
        Linear(int input_dim, int output_dim) {
            W = randn(input_dim, output_dim).set_grad();
            b = randn(1, output_dim).set_grad();
        }

        Tensor<double> forward(const Tensor<double> &a){
            if(a.get_dim() == 1){
                a.unsqueeze();
            }
            return a.cross(W) + b;
        }

        void print_grad(){
            std::cout << W.grad() << std::endl << b.grad() << std::endl;
        }

        void zero_grad(){
            W.zero_grad();
            b.zero_grad();
        }

        std::vector<std::tuple<double *, double *, int>> parameters(){
            std::vector<std::tuple<double *, double *, int>> ret;
            ret.push_back(std::tuple<double*, double*, int>{W.data.get(), W.graddata.get(), W.get_size()});
            ret.push_back(std::tuple<double*, double*, int>{b.data.get(), b.graddata.get(), b.get_size()});
            return ret;
        }
    };
}

#endif //CLIONPROJECT_LAYER_H
