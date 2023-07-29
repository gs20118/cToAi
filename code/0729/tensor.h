//
// Created by 최정환 on 2023-07-29.
//

#ifndef NEWPROJECTS_TENSOR_H
#define NEWPROJECTS_TENSOR_H
#include "cailib.h"
#include "range.h"
#include "operator.h"
#include <random>
#include <memory>
#include <functional>
#include <initializer_list>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <map>

namespace func{
    template<typename T>
    std::shared_ptr<T> newArray(int n, std::shared_ptr<T> arr)
    {
        std::shared_ptr<T> ret = std::shared_ptr<T>(new T[n], [](T* a){delete[] a;});
        for(int i=0; i<n; i++) ret.get()[i] = arr[i];
        return ret;
    }

    template<typename T>
    T* newArray_(int n, T* arr){
        T* ret = new T[n];
        for(int i=0; i<n; i++) ret[i] = arr[i];
        return ret;
    }

    template<typename T>
    std::shared_ptr<T> newArray(int n, T val)
    {
        std::shared_ptr<T> ret = std::shared_ptr<T>(new T[n], [](T* a){delete[] a;});
        for(int i=0; i<n; i++) ret.get()[i] = val;
        return ret;
    }

    template<typename T>
    T* newArray_(int n, T val)
    {
        T* ret = new T[n];
        for(int i=0; i<n; i++) ret[i] = val;
        return ret;
    }

    template<typename T>
    std::shared_ptr<T> newArray(std::vector<T> arr)
    {
        std::shared_ptr<T> ret = std::shared_ptr<T>(new T[arr.size()], [](T* a){delete[] a;});
        for(int i=0; i<arr.size(); i++) ret.get()[i] = arr[i];
        return ret;
    }

    template<typename T>
    T* newArray_(std::vector<T> arr)
    {
        T* ret = new T[arr.size()];
        for(int i=0; i<arr.size(); i++) ret[i] = arr[i];
        return ret;
    }

    std::string multiple(std::string str, int n){
        std::string ret = "";
        while(n--) ret += str;
        return ret;
    }
}
namespace cai {
    template<typename T>
    class Tensor {
    private:
        std::shared_ptr<T> data = nullptr;
        bool indexed = false;
        int dim{};
        int offset{};
        int *stride{};
        int *shape{};
        bool conjugate = true;

    public:
        bool requires_grad = false;
        std::shared_ptr<T> graddata = nullptr;
        Operator<T> *grad_func = nullptr;

        //생성자들
        Tensor(){
            this->dim = 0;
            data = std::shared_ptr<T>(new T[1], [](T* a){delete[] a;});
            stride = nullptr;
            shape = nullptr;
            offset = 0;
            conjugate = true;
        }
        ~Tensor(){
            delete[] stride;
            delete[] shape;
            data.reset();
        }
        Tensor(T v, bool grad=false){
            data.get()[0] = v;
            if(grad) set_grad();
        }

        Tensor(const std::initializer_list<int> &sh, T val = 0, bool grad = false)
        :Tensor(std::vector<int>(sh.begin(), sh.end()), val, grad) {
            if(grad) set_grad();
        }
        Tensor(const std::initializer_list<T> &val, const std::initializer_list<int> &sh, bool grad = false)
        :Tensor(std::vector<T>(val.begin(), val.end()), std::vector<int>(sh.begin(), sh.end())) {
            if(grad) set_grad();
        }

        Tensor(const std::vector<int> &sh, T val = 0, bool grad = false){
            int cnt = 0;
            dim  = sh.size();
            shape = new int[dim];
            stride = new int[dim];
            offset = 0;

            for(const int & x : sh){
                shape[cnt] = x;
                cnt++;
            }

            stride[dim-1] = 1;
            for(int i=dim-2; i>=0; i--)
                stride[i] = stride[i+1] * shape[i+1];

            data = func::newArray(get_size(), val);
            conjugate = true;
            if(grad) set_grad();
        }
        Tensor(const std::vector<T> &val, const std::vector<int> &sh, bool grad = false)
        :Tensor(sh, 0 , grad) {
            int cnt = 0, num = get_size();
            if (num != val.size()) {
                throw std::range_error("you can't make " + std::to_string(sh.size()) + " to " + std::to_string(num) + "\n");
            }
            for (const T &x: val) {
                data.get()[cnt] = x;
                cnt++;
            }
        }

        Tensor(int dim, int *shape, T val = 0, bool grad = false){
            this->dim = dim;
            this->offset = 0;
            this->stride = new int[this->dim];
            this->shape = shape;
            for(int i =this->dim-1 ; i >=0 ; i--) {
                if(i==this->dim-1) stride[i] = 1;
                else stride[i] = stride[i+1] * this->shape[i+1];
            }
            conjugate = true;
            data = func::newArray<T>(get_size(), val);
            if(grad) set_grad();
        }
        Tensor(int dim, int *stride, int *shape, std::shared_ptr<T> data, int offset = 0,
               std::shared_ptr<T> graddata = nullptr, Operator<T>* grad_func= nullptr){
            this->dim = dim;
            this->stride = stride;
            this->shape = shape;
            set_conj();
            this->data = data;
            this->offset = offset;
            if(graddata != nullptr){
                requires_grad = true;
                this->graddata = graddata;
                this->grad_func = grad_func;
            }
        }
        Tensor(const Tensor &o)
        :Tensor(o.dim,func::newArray_(o.dim, o.stride),
                func::newArray_(o.dim, o.shape),o.data, o.offset, o.graddata, o.grad_func){}

        //set, get
        void set_grad(){
            requires_grad = true;
            graddata = func::newArray<T>(get_size(), 0);
        }
        void set_conj(){
            int chk = 1;
            for (int i = dim-1; i>=0; i--)
            {
                if(chk != stride[i]) {
                    conjugate = false;
                    return;
                }
                chk *= shape[i];
            }
        }
        void set(const Tensor &o){
            if(!sameShape(o)) throw std::length_error("the shape is not equal");
            std::vector<int> v = initi();
            int num = get_size();
            for(int i=0; i<num; i++) {
                nexti(v);
                item(v) = o.item(v);
            }
        }
        void set(T val){
            std::vector<int> v = initi();
            int num = get_size();
            for(int i=0; i<num; i++) {
                nexti(v);
                item(v) = val;
            }
        }
        //void set_grad();

        int get_size() const{
            int ret = 1;
            for(int i=0; i<dim;i++)
                ret *= shape[i];
            return ret;
        }

        T &item() const{
            if(dim!=0){
                throw std::domain_error("Can't get the item if dimension isn't 0");
            }
            return data.get()[0];
        }
        T &item(const std::vector<int> &pos) const{
            return data.get()[index(pos)];
        }
        template<typename... Ints>
        T &item(Ints... b) const{
            std::vector<int> idx{b...};
            return item(idx);
        }

        Tensor get(const std::vector<cai::Range> &pos) const{
            if(pos.size() > dim){
                throw std::range_error("index is over " + std::to_string(dim) + "\n");
            }

            int newDim = dim;
            for(const Range & p : pos){
                if(p.v) newDim--;
            }

            int *newShape = new int[newDim];
            int *newStride = new int[newDim];
            int newOffset = offset;
            int cnt = 0;

            for(int i=0; i<dim; i++){
                if(i<pos.size()){
                    newOffset += pos[i].s * stride[i];
                    if(!pos[i].v){
                        newShape[cnt] = (pos[i].e == -1  ? shape[i]:pos[i].e) - pos[i].s;
                        newStride[cnt] = stride[i];
                        cnt++;
                    }
                }
                else{
                    newShape[cnt] = shape[i];
                    newStride[cnt] = stride[i];
                    cnt++;
                }
            }

            Tensor ret = Tensor<T>(newDim, newStride, newShape, data, newOffset);
            return ret;
        }
        template<typename... Ranges>
        Tensor get(Ranges... b) const{
            return get(std::vector<cai::Range>{b...});
        }
        template<typename... Ranges>
        Tensor operator[](Ranges... b) const{
            cai::Tensor<T> ret =  get(b...);
            ret.indexed = true;
            return ret;
        }

        void nexti(std::vector<int> &v) const{
            if(dim>=1) v[dim-1] ++;
            int n = dim-1;
            while(n>=1){
                if(v[n] < shape[n]) break;
                v[n] = 0 ;
                v[n-1] ++;
                n--;
            }
        }
        std::vector<int> initi() const{
            std::vector<int> ret(dim);
            if(dim>=1) ret[dim-1] = -1;
            return ret;
        }

        Tensor reshape(const std::vector<int>& resh) const{
            if(!conjugate){
                throw std::domain_error( "you can't reshape, the tensor is not conjugate \n");
            }

            int num = 1;
            for(const int &x : resh) num *= x;

            if(get_size() != num){
                throw std::range_error( "you can't reshape, the number of item is different \n");
            }

            int newDim = resh.size();
            int* newShape = func::newArray_(resh);
            int* newStride = new int[newDim];
            int cnt = 1;
            for(int i=newDim-1; i>=0; i--){
                if(i != newDim-1) cnt *= newShape[i+1];
                newStride[i] = cnt;
            }

            Tensor<T> ret(newDim , newStride, newShape, data);
            return ret;
        }
        template<typename... Ints>
        Tensor reshape(Ints... b) const{
            std::vector<int> idx{b...};
            return reshape(idx);
        }

        Tensor copy() const{
            int newDim = dim;
            int* newShape = func::newArray_(dim, shape);
            int* newStride = new int[newDim];
            int cnt = 1;
            for(int i=newDim-1; i>=0; i--){
                newStride[i] = cnt;
                cnt *= newShape[i];
            }

            std::shared_ptr<T> newData =  std::shared_ptr<T>(new T[cnt], [](T* a){delete[] a;});
            std::vector<int> pos = initi();
            for(int i=0; i<cnt; i++){
                nexti(pos);
                newData.get()[i] = item(pos);
            }
            Tensor<T> ret(newDim , newStride, newShape,newData, 0);
            return ret;
        }
        Tensor grad(){
            return Tensor<T>(dim, func::newArray_(dim, stride), func::newArray_(dim, shape), graddata, offset);
        }

        int index(const std::vector<int> &pos) const{
            int idx = offset;
            if(pos.size() != dim){
                throw std::domain_error("there isn't index for indicator which dimension isn't equal to tensor\n"
                                        + std::to_string(pos.size()) + " is not equal to " + std::to_string(dim) + "\n");
            }
            for(int i=0; i<dim; i++){
                if(pos[i]>=shape[i]){
                    throw std::range_error("index over shape in dim:" + std::to_string(dim) + "\n");
                }
                idx += pos[i] * stride[i];
            }
            return idx;
        }
        template<typename... Ints>
        int index(Ints... b) const{
            std::vector<int> idx{b...};
            return index(idx);
        }

        //출력
        void print() const{
            print(std::cout);
        }
        void print_all() const{
            std::cout << "dim : " <<  dim << std::endl;
            std::cout << "shape : (" ;
            for(int i=0; i<dim; i++){
                std::cout<< shape[i] ;
                if(i!=dim -1) std:: cout<< ", ";
                else std::cout<<")\n";
            }
            std::cout << "stride : (" ;
            for(int i=0; i<dim; i++){
                std::cout<< stride[i] ;
                if(i!=dim -1) std:: cout<< ", ";
                else std::cout<<")\n";
            }
            std::cout << "offset : " << offset << std::endl;
            std::cout << "conjugate : " << conjugate << std::endl;
        }
        void print(std::ostream &o) const{
            o<<toString();
        }
        std::string toString() const{
            if(dim==0) return std::to_string(data.get()[0]);
            std::vector<int> temp = initi();
            std::string ret = "";
            int num = get_size();

            int n = 0;
            while(num--){
                nexti(temp);
                int len = std::to_string(item(temp)).size();
                n = std::min(10, std::max(n,  len));
            }

            temp = initi();
            num = get_size();
            while(num--){
                nexti(temp);
                int cnt = 0;
                while(cnt <= dim-1 && temp[dim-1-cnt] == 0) cnt++;
                if(cnt>0){
                    ret += func::multiple(" " , (dim-cnt));
                    ret += func::multiple("[" , cnt);
                }

                std::string str = std::to_string(item(temp));
                if(str.size() < n) ret += func::multiple(" ", (n-str.size()));
                if(str.size() > n) str = str.substr(n);
                ret  += str;

                cnt = 0;
                while( cnt <= dim-1 && temp[dim-1 - cnt] == shape[dim-1 - cnt]-1) cnt++;
                if(cnt>0) ret += func::multiple("]" , cnt);
                if(cnt != dim) ret += ",";
                if(cnt>0) ret += "\n";
            }
            return ret;
        }

        friend std::ostream& operator<<( std::ostream& o, const Tensor& t){
            t.print(o);
            return o;
        }

        Tensor &operator=(const Tensor &o) {
            if (indexed){
                set(o);
            }
            else{
                dim = o.dim;
                shape = func::newArray_(dim, o.shape);
                stride = func::newArray_(dim, o.stride);
                conjugate = o.conjugate;
                data = o.data;
                offset = o.offset;
                graddata = o.graddata;
                requires_grad = o.requires_grad;
                grad_func = o.grad_func;
            }
        }
        Tensor &operator=(T v){
            if (indexed){
                set(v);
            }
            else{
                dim = 0;
                shape = nullptr;
                stride = nullptr;
                conjugate = true;
                data = func::newArray(1, v);
                offset = 0;
            }
        }
        bool sameShape(const Tensor &o) const{
            if(dim != o.dim){
                throw std::domain_error("dimension isn't same");
                return false;
            }
            for(int i=0; i<dim; i++){
                if(shape[i] != o.shape[i]){
                    throw std::domain_error("shape isn't same");
                    return false;
                }
            }
            return true;
        }
        void foreach(std::function<void(Tensor&, std::vector<int>&)> func){
            auto v = initi();
            int num = get_size();
            while(num--){
                nexti(v);
                func(*this, v);
            }
        }

        Tensor operator+(const Tensor &o){
            Operator<T> * op = new Add<T>();
            Tensor<T> ret = op->forward(*this, o);
            if(requires_grad || o.requires_grad) ret.grad_func = op;
            return ret;
        }


        //텐서 생성용
        template<typename U>
        friend Tensor<U> zeros_like(const Tensor<U> &o);
        template<typename U>
        friend Tensor<U> ones_like(const Tensor<U> &o);
        template<typename U, typename... Ints>
        friend Tensor<U> arange(Ints... v);
        template<typename... Ints>
        friend Tensor<double> rand(Ints... v);
        template<typename... Ints>
        friend Tensor<double> randn(Ints... v);
        template<typename... Ints>
        friend Tensor<int> randint(int low, int high, Ints... v);
        //==============================================
    };

    template<typename U>
    Tensor<U> zeros_like(const Tensor<U> &o) {
        return Tensor<U>(o.dim, o.shape, 0);
    }
    template<typename U>
    Tensor<U> ones_like(const Tensor<U> &o) {
        return Tensor<U>(o.dim, o.shape, 1);
    }
    template<typename U, typename... Ints>
    Tensor<U> arange(Ints... v) {
        Tensor<U> ret({v...}, 0);
        auto vec = ret.initi();
        int num = ret.get_size();
        for (int i = 0; i < num; i++) {
            ret.nexti(vec);
            ret.item(vec) = i;
        }
        return ret;
    }
    template<typename... Ints>
    Tensor<double> rand(Ints... v) {
        Tensor<double> ret({v...}, 0);
        auto vec = ret.initi();
        int num = ret.get_size();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(0, 1);
        for (int i = 0; i < num; i++) {
            ret.nexti(vec);
            ret.item(vec) = dis(gen);
        }
        return ret;
    }
    template<typename... Ints>
    Tensor<double> randn(Ints... v) {
        Tensor<double> ret({v...}, 0);
        auto vec = ret.initi();
        int num = ret.get_size();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dis(0, 1);
        for (int i = 0; i < num; i++) {
            ret.nexti(vec);
            ret.item(vec) = dis(gen);
        }
        return ret;
    }
    template<typename... Ints>
    Tensor<int> randint(int low, int high, Ints... v) {
        Tensor<int> ret({v...}, 0);
        auto vec = ret.initi();
        int num = ret.get_size();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(low, high);
        for (int i = 0; i < num; i++) {
            ret.nexti(vec);
            ret.item(vec) = dis(gen);
        }
        return ret;
    }
}

#endif //NEWPROJECTS_TENSOR_H