//
// Created by 최정환 on 2023-07-29.
//

#ifndef NEWPROJECTS_RANGE_H
#define NEWPROJECTS_RANGE_H
#include <iostream>
#include "cailib.h"
#define r(a, b) cai::Range(a, b)

namespace cai{
    class Range{
    public:
        int s, e;
        bool v = false;
        Range() : s(0), e(-1) {}
        Range(int x) : s(x), e(x) {
            v = true;
        }
        Range(int s, int e) {
            this->s = s;
            this->e = e;
        }

        void print() const{
            print(std::cout);
        }
        void print(std::ostream &o) const{
            o<<"range(" + std::to_string(s) + ":"+ std::to_string(e) + ")\n";
        }
    };
}
std::ostream& operator<<(std::ostream& o, const cai::Range& r) {
    r.print(o);
    return o;
}


#endif //NEWPROJECTS_RANGE_H
