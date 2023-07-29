#include "tensor.h"

/**** Tensor 생성자/소멸자 정의 ****/



//dim = 0 용
template<typename T>
cai::Tensor<T>::Tensor() {
    this->dim = 0;
    data = std::make_shared<T>(1);
    stride = nullptr;
    shape = nullptr;
    offset = 0;
    conjugate = true;
}

template<typename T>
cai::Tensor<T>::~Tensor() {
    delete[] stride;
    delete[] shape;
    data.reset();
}

template<typename T>
cai::Tensor<T>::Tensor(T v)
        :Tensor(){
    data.get()[0] = v;
}



//사용자 용
template<typename T>
cai::Tensor<T>::Tensor(const std::initializer_list<int>& sh, T val)
        :Tensor(std::vector<int>(sh.begin(), sh.end()), val) {}

template<typename T>
cai::Tensor<T>::Tensor(const std::initializer_list<T>& val, const std::initializer_list<int>& sh)
        :Tensor(std::vector<T>(val.begin(), val.end()), std::vector<int>(sh.begin(), sh.end())) {}



//내부 구현 용


template<typename T>
cai::Tensor<T>::Tensor(int dim, int *stride, int *shape, std::shared_ptr<T> data, int offset)
{
    this->dim = dim;
    this->stride = stride;
    this->shape = shape;
    set_conj();
    this->data = data;
    this->offset = offset;
}

template<typename T>
cai::Tensor<T>::Tensor(int dim, int *shape, T val){
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
}

template<typename T>
cai::Tensor<T>::Tensor(const Tensor &o)
        :Tensor(o.dim,func::newArray(o.dim, o.stride),
                func::newArray(o.dim, o.shape),o.data, o.offset){}

template<typename T>
cai::Tensor<T>::Tensor(const std::vector<int>& sh, T val)
{
    int cnt = 0, num;
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
}

template<typename T>
cai::Tensor<T>::Tensor(const std::vector<T>& val, const std::vector<int>& sh)
        :Tensor(sh) {
    int cnt = 0, num = get_size();
    if (num != val.size()) {
        throw std::range_error("you can't make " + std::to_string(sh.size()) + " to " + std::to_string(num) + "\n");
    }
    for (const T &x: val) {
        data.get()[cnt] = x;
        cnt++;
    }
}




//==========================================================================

/**** Tensor Operation 정의 ****/



template<typename T>
cai::Tensor<T>& cai::Tensor<T>::operator=(const cai::Tensor<T> &o) {
    if(!indexed){
        dim = o.dim;
        shape = func::newArray(dim, o.shape);
        stride = func::newArray(dim, o.stride);
        offset = o.offset;
        conjugate = o.conjugate;
        data = o.data;
    }
    else{
        set(o);
    }
    return *this;
}

template<typename T>
cai::Tensor<T>& cai::Tensor<T>::operator=(T val) {
    if(indexed == false){
        this->dim = 0;
        data = func::newArray<T>(1, val);
        stride = nullptr;
        shape = nullptr;
        offset = 0;
        conjugate = true;
    }
    else{
        set(val);
    }
    return *this;
}

template<typename T>
bool cai::Tensor<T>::sameShape(const Tensor<T> &o) const{
    if( dim!=o.dim ) return false;
    for(int i=0; i<dim; i++) if(shape[i]!=o.shape[i]) return false;
    return true;
}

template<typename T>
void cai::Tensor<T>::foreach(std::function<void(Tensor&, std::vector<int>&)> func) {
    auto vec = initi();
    int num = get_size();
    while (num--) {
        nexti(vec);
        func((*this), vec);
    }
}




//==========================================================================

/**** Tensor setter, getter 정의 ****/



//set
template<typename T>
void cai::Tensor<T>::set_conj() {
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

template<typename T>
void cai::Tensor<T>::set(const Tensor<T> &o)
{
    if(!sameShape(o)) throw std::length_error("the shape is not equal");
    std::vector<int> v = initi();
    int num = get_size();
    for(int i=0; i<num; i++) {
        nexti(v);
        item(v) = o.item(v);
    }
}

template<typename T>
void cai::Tensor<T>::set(T val){
    std::vector<int> v = initi();
    int num = get_size();
    for(int i=0; i<num; i++) {
        nexti(v);
        item(v) = val;
    }
}

template<typename  T>
int cai::Tensor<T>::get_size() const{
    int ret = 1;
    for(int i=0; i<dim;i++)
        ret *= shape[i];
    return ret;
}

//item
template<typename T>
T& cai::Tensor<T>::item()const{
    if(dim!=0){
        throw std::domain_error("Can't get the item if dimension isn't 0");
    }
    return data.get()[0];
}

template<typename T>
T& cai::Tensor<T>::item(const std::vector<int>& pos) const{
    return data.get()[index(pos)];
}

template<typename T>
template<typename... Ints>
T& cai::Tensor<T>::item(Ints... b)const{
    std::vector<int> idx{b...};
    return item(idx);
}



//index
template<typename T>
int cai::Tensor<T>::index(const std::vector<int>& pos) const{
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

template<typename T>
template<typename... Ints>
int cai::Tensor<T>::index(Ints... b) const{
    std::vector<int> idx{b...};
    return index(idx);
}

template<typename T>
void cai::Tensor<T>::nexti(std::vector<int>& v) const{
    if(dim>=1) v[dim-1] ++;
    int n = dim-1;
    while(n>=1){
        if(v[n] < shape[n]) break;
        v[n] = 0 ;
        v[n-1] ++;
        n--;
    }
}

template<typename T>
std::vector<int> cai::Tensor<T>::initi() const{
    std::vector<int> ret(dim);
    if(dim>=1) ret[dim-1] = -1;
    return ret;
}


//get
template<typename T>
cai::Tensor<T> cai::Tensor<T>::get(const std::vector<Range>& pos) const{
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

template<typename T>
template<typename... Ranges>
cai::Tensor<T> cai::Tensor<T>::get(Ranges... b)const{
    return get(std::vector<cai::Range>{b...});
}

template<typename T>
template<typename... Ranges>
cai::Tensor<T> cai::Tensor<T>::operator[](Ranges... b) const{
    cai::Tensor<T> ret =  get(b...);
    ret.indexed = true;
    return ret;
}


//reshape
template<typename T>
cai::Tensor<T> cai::Tensor<T>::reshape(const std::vector<int>& resh) const{
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

template<typename T>
template<typename... Ints>
cai::Tensor<T> cai::Tensor<T>::reshape(Ints... b) const{
    std::vector<int> idx{b...};
    return reshape(idx);
}


template<typename T>
cai::Tensor<T> cai::Tensor<T>::copy() const{
    int newDim = dim;
    int* newShape = func::newArray_(dim, shape);
    int* newStride = new int[newDim];
    int cnt = 1;
    for(int i=newDim-1; i>=0; i--){
        newStride[i] = cnt;
        cnt *= newShape[i];
    }

    std::shared_ptr<T> newData = std::make_shared<T>(cnt);
    std::vector<int> pos = initi();
    for(int i=0; i<cnt; i++){
        nexti(pos);
        newData.get()[i] = item(pos);
    }
    Tensor<T> ret(newDim , newStride, newShape,newData, 0);
    return ret;
}

//==========================================================================

/**** Tensor 출력 정의 ****/



template<typename T>
void cai::Tensor<T>::print() const {
    print(std::cout);
}

template<typename T>
void cai::Tensor<T>::print_all()const {
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
}

template<typename T>
void cai::Tensor<T>::print(std::ostream &o) const {
    o<<toString();
}

template<typename T>
std::string cai::Tensor<T>::toString_(std::vector<int> pos) const {
    if(pos.size() == dim){
        return std::to_string(item(pos));
    }

    std::string ret = "[";
    int ldim = dim - pos.size();
    for(int i=0; i<shape[pos.size()]; i++){
        std::vector<int> temp(pos);
        temp.push_back(i);
        ret += toString_(temp);
        if( i!= shape[pos.size()]-1) {
            ret += ", ";
            if (ldim >= 2) ret += "\n";
            if (ldim >= 3) ret += "\n";
        }
    }
    ret += "]";

    return ret;
}

template<typename T>
std::string cai::Tensor<T>::toString() const {
    std::vector<int> temp;
    return toString_(temp);
}


//==========================================================================



namespace cai {
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

