/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef STACK_UTILS_H
#define STACK_UTILS_H

//stack utils
namespace FT{
    void dev_allocate(float * f, bool * b, size_t Sizef, size_t Sizeb);
    void copy_from_device(float * dev_f, float * host_f, bool * dev_b, bool * host_b, size_t Sizef,
                            size_t Sizeb);
    void free_device(float * dev_f, bool * dev_b);
}
#endif
