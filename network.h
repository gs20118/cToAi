//
// Created by choijunghwan on 2023-08-01.
//

#ifndef CLIONPROJECT_NETWORK_H
#define CLIONPROJECT_NETWORK_H
#include "layer.h"
#include <vector>
namespace cai{
    class Network{
    public:
        std::map<std::string, Layer*> layers;

        ~Network(){
            for(auto &layer : layers){
                delete layer.second;
            }
        }

        void set(std::string str, Layer *layer){
            layers.insert({str, layer});
        }

        virtual void init(){}

        virtual Tensor<double> forward(Tensor<double> &a){}

        Tensor<double> operator()(Tensor<double> &a){
            return forward(a);
        }

        void zero_grad(){
            for(auto &layer : layers){
                layer.second->zero_grad();
            }
        }

        std::vector<std::tuple<double*, double*, int>> parameters(){
            std::vector<std::tuple<double*, double*, int>> ret;
            for(auto &layer : layers){
                auto para = layer.second->parameters();
                for(auto &tup : para){
                    ret.push_back(tup);
                }
            }
            return ret;
        }
    };
}
#endif //CLIONPROJECT_NETWORK_H
