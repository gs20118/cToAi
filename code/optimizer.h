//
// Created by choijunghwan on 2023-08-01.
//

#ifndef CLIONPROJECT_OPTIMIZER_H
#define CLIONPROJECT_OPTIMIZER_H
#include<vector>

template<typename T>
class Optimizer{
public:
    std::vector<T*> data;
    std::vector<T*> grad;

    virtual void optim(){};
    virtual void set(std::vector<std::tuple<T*, T*, int>> vec){}
    virtual void init(){}
};

template<typename T>
class SimpleOptim : public Optimizer<T>{
public:
    double lr;
    int num;

    SimpleOptim(double lr=0.01){
        this->lr = lr;
        this->num = 0;
    }

    void optim(){
        for(int i=0; i<num; i++){
            T* data = Optimizer<T>::data[i];
            T* grad = Optimizer<T>::grad[i];
            (*data) -= lr * (*grad);
        }
    };

    void set(std::vector<std::tuple<T*, T*, int>> vec){
        for(auto &tup : vec){
            int cnt = std::get<2>(tup);
            num += std::get<2>(tup);

            for(int i=0; i<cnt; i++) {
                Optimizer<T>::data.push_back(std::get<0>(tup) + i);
                Optimizer<T>::grad.push_back(std::get<1>(tup) + i);
            }
        }
    }
};

#endif //CLIONPROJECT_OPTIMIZER_H
