#ifndef CLIONPROJECT_TENSOR_H
#define CLIONPROJECT_TENSOR_H

#include "cailib.h"
#include "range.h"
#include <random>
#include <memory>



namespace func{
    template<typename T>
    std::shared_ptr<T> newArray(int n, std::shared_ptr<T> arr)
    {
        std::shared_ptr<T> ret = std::make_shared<T>(n);
        for(int i=0; i<n; i++) ret[i] = arr[i];
        return ret;
    }

    template<typename T>
    T* newArray_(int n, T* arr)
    {
        T* ret = new T[n];
        for(int i=0; i<n; i++) ret[i] = arr[i];
        return ret;
    }

    template<typename T>
    std::shared_ptr<T> newArray(int n, T val)
    {
        std::shared_ptr<T> ret = std::make_shared<T>(n);
        for(int i=0; i<n; i++) ret[i] = val;
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
        std::shared_ptr<T> ret = std::make_shared<T>(arr.size());
        for(int i=0; i<arr.size(); i++) ret[i] = arr[i];
        return ret;
    }

    template<typename T>
    T* newArray_(std::vector<T> arr)
    {
        T* ret = new T[arr.size()];
        for(int i=0; i<arr.size(); i++) ret[i] = arr[i];
        return ret;
    }
}
namespace cai {
    template<typename T>
    class Tensor {
    private:
        std::shared_ptr<T> data = nullptr;
        bool indexed = false;
        int dim;
        int offset;
        int *stride;
        int *shape;
        bool conjugate = true;

    public:
        bool requires_grad = false;
        Tensor<T> *grad = nullptr;
        //Operation::Operator *grad_fun = nullptr ;

        //생성자들
        Tensor();
        ~Tensor();
        Tensor(T v);
        Tensor(const std::initializer_list<int> &sh, T val = 0);
        Tensor(const std::initializer_list<T> &val, const std::initializer_list<int> &sh);
        Tensor(const std::vector<int> &sh, T val = 0);
        Tensor(const std::vector<T> &val, const std::vector<int> &sh);
        Tensor(int dim, int *shape, T val = 0);
        Tensor(int dim, int *stride, int *shape, std::shared_ptr<T> data, int offset = 0);
        Tensor(const Tensor &other);

        //set, get
        void set_conj();
        void set(const Tensor &o);
        void set(T v);
        //void set_grad();

        int get_size() const;

        T &item() const;
        T &item(const std::vector<int> &pos) const;
        template<typename... Ints>
        T &item(Ints... b) const;

        Tensor get(const std::vector<cai::Range> &pos) const;
        template<typename... Ranges>
        Tensor get(Ranges... b) const;

        template<typename... Ranges>
        Tensor operator[](Ranges... b) const;

        void nexti(std::vector<int> &v) const;
        std::vector<int> initi() const;

        Tensor reshape(const std::vector<int>& resh) const;

        template<typename... Ints>
        Tensor reshape(Ints... b) const;

        Tensor copy() const;

        int index(const std::vector<int> &pos) const;
        template<typename... Ints>
        int index(Ints... b) const;

        //출력
        void print() const;
        void print_all() const;
        void print(std::ostream &o) const;
        std::string toString() const;
        std::string toString_(std::vector<int> pos) const;


        friend std::ostream& operator<<( std::ostream& o, const Tensor& t){
            t.print(o);
            return o;
        }

        Tensor &operator=(const Tensor &o);
        Tensor &operator=(T v);
        Tensor operator+(const Tensor &o) const;
        Tensor adfunc(const Tensor &o, T(*func(T, T)) ) const;
        Tensor adfunc(T(*func(T)) ) const;
        Tensor adfunc(T(*func(T, T)) , int axis=-1) const;

        bool sameShape(const Tensor &o) const;

        void foreach(std::function<void(Tensor&, std::vector<int>&)> func);\


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
}






#endif
