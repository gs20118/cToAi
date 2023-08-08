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

template<typename T>
class AdamOptim : public Optimizer<T>{
public:
    double lr, b1, b2, e;
    std::vector<T> moment;
    std::vector<T> grad_ac;

    int num, t;

    AdamOptim(double lr=0.01, double b1=0.9, double b2 = 0.99, double e = pow(0.1, 8)){
        this->lr = lr;
        this->b1 = b1;
        this->b2 = b2;
        this->e = e;
        this->num = 0;
        t = 0;
    }

    void optim(){
        t++;
        for(int i=0; i<num; i++){

            T* data = Optimizer<T>::data[i];
            T* grad = Optimizer<T>::grad[i];
            moment[i] = b1*moment[i] + (1-b1)*(*grad);
            grad_ac[i] = b1*grad_ac[i] + (1-b1)*pow((*grad), 2);

            T mt = moment[i] / (1 - pow(b1, t));
            T gt = grad_ac[i] / (1-pow(b2, t));
            (*data) -= lr * (mt) / sqrt(gt + e);
        }

    };

    void set(std::vector<std::tuple<T*, T*, int>> vec){
        for(auto &tup : vec){
            int cnt = std::get<2>(tup);
            num += std::get<2>(tup);

            for(int i=0; i<cnt; i++) {
                Optimizer<T>::data.push_back(std::get<0>(tup) + i);
                Optimizer<T>::grad.push_back(std::get<1>(tup) + i);
                moment.push_back(0);
                grad_ac.push_back(0);
            }
        }
    }
};

#endif //CLIONPROJECT_OPTIMIZER_H
