/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "state.h"
#include <iostream>

namespace FT
{
    namespace Dat{
    
    #ifndef USE_CUDA
    
        bool State::check(std::map<char, unsigned int> &arity)
        {
            if(arity.find('z') == arity.end())
                return (f.size() >= arity['f'] &&
                        b.size() >= arity['b'] &&
                        c.size() >= arity['c']);
            else
                return (f.size() >= arity['f'] &&
                        b.size() >= arity['b'] &&
                        c.size() >= arity['c'] &&
                        z.size() >= arity['z']);
        }
        
        ///< checks if arity of node provided satisfies the node names in various string State
        bool State::check_s(std::map<char, unsigned int> &arity)
        {
            if(arity.find('z') == arity.end())
                return (fs.size() >= arity['f'] && 
                        bs.size() >= arity['b'] &&
                        cs.size() >= arity['c']);
            else
                return (fs.size() >= arity['f'] &&
                        bs.size() >= arity['b'] &&
                        cs.size() >= arity['c'] &&
                        zs.size() >= arity['z']);
        }
        
    #else
        State::State()
        {
            idx['f'] = 0;
            idx['c'] = 0;
            idx['b'] = 0;
        }
        
        void State::update_idx(char otype, std::map<char, unsigned>& arity)
        {
            ++idx[otype];
            for (const auto& a : arity)
                    idx[a.first] -= a.second;
        }
        
        bool State::check(std::map<char, unsigned int> &arity)
        {
            if(arity.find('z') == arity.end())
                return (f.rows() >= arity['f'] && 
                        c.rows() >= arity['c'] &&
                        b.rows() >= arity['b']);
            else
                return (f.rows() >= arity['f'] &&
                        c.rows() >= arity['c'] &&
                        b.rows() >= arity['b'] &&
                        z.size() >= arity['z']);
        }
        
        bool State::check_s(std::map<char, unsigned int> &arity)
        {
            if(arity.find('z') == arity.end())
                return (fs.size() >= arity['f'] &&
                        cs.size() >= arity['c'] &&
                        bs.size() >= arity['b']);
            else
                return (fs.size() >= arity['f'] &&
                        cs.size() >= arity['c'] &&
                        bs.size() >= arity['b'] &&
                        zs.size() >= arity['z']);
        }
        
        void State::allocate(const std::map<char, size_t>& stack_size, size_t N)
        {
            //std::cout << "before dev_allocate, dev_f is " << dev_f << "\n";
            dev_allocate(dev_f, N*stack_size.at('f'),
                         dev_c, N*stack_size.at('c'),
                         dev_b, N*stack_size.at('b'));
            //std::cout << "after dev_allocate, dev_f is " << dev_f << "\n";

	        //printf("Allocated Stack Sizes\n");
	        //printf("\tFloating stack N=%zu and stack size as %zu\n",N, stack_size.at('f'));

            this->N = N;
	
            f.resize(stack_size.at('f'),N);
            c.resize(stack_size.at('c'),N);
            b.resize(stack_size.at('b'),N);
        }

        void State::limit()
        {
            // clean floating point stack. 
            for (unsigned r = 0 ; r < f.rows(); ++r)
            {
                f.row(r) = (isinf(f.row(r))).select(MAX_DBL,f.row(r));
                f.row(r) = (isnan(f.row(r))).select(0,f.row(r));
            }
            
            for (unsigned r = 0 ; r < c.rows(); ++r)
            {
                c.row(r) = (isinf(c.row(r))).select(MAX_INT, c.row(r));
                c.row(r) = (isnan(c.row(r))).select(0, c.row(r));
            }
        }
        
        /// resize the f and b State to match the outputs of the program
        void State::trim()
        {
            //std::cout << "resizing f to " << idx['f'] << "x" << f.cols() << "\n";
            //f.resize(idx['f'],f.cols());
            //b.resize(idx['b'],b.cols());
            //std::cout << "new f size: " << f.size() << "," << f.rows() << "x" << f.cols() << "\n";
            //unsigned frows = f.rows()-1;
            //for (unsigned r = idx['f']; r < f.rows(); ++r)
            //    f.block(r,0,frows-r,f.cols()) = f.block(r+1,0,frows-r,f.cols());
            //    f.conservativeResize(frows,f.cols());
            f.conservativeResize(idx['f'], f.cols());
            c.conservativeResize(idx['c'], c.cols());
            b.conservativeResize(idx['b'], b.cols());
        }
        
        void State::copy_to_host()
        {
            /* std::cout << "size of f before copy_from_device: " << f.size() */ 
            /*           << ", stack size: " << N*stack_size.at('f') << "\n"; */
            /* std::cout << "size of b before copy_from_device: " << b.size() */ 
            /*           << ", stack size: " << N*stack_size.at('b') << "\n"; */
	     
            copy_from_device(dev_f, f.data(), N*idx['f'],
                             dev_c, c.data(), N*idx['c'],
                             dev_b, b.data(), N*idx['b']);
            //copy_from_device(dev_f, f.data(), dev_b, b.data(), N*stack_size.at('f'), N*stack_size.at('b'));

            trim(); 
            limit();
        }
        
        State::~State()
        {
            free_device(dev_f, dev_c, dev_b);
        }

    #endif
    }
}

