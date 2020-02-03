/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "state.h"
#include <iostream>

#ifdef USE_CUDA
    #include "../pop/op/node.h"
#endif

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
        
        ///< checks if arity of node provided satisfies the node names in 
        // various string State
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
        
        void Trace::copy_to_trace(State& state, std::map<char, 
                unsigned int> &arity)
        {
            for (int i = 0; i < arity['f']; i++) {
                /* cout << "push back float arg for " << program.at(i)->name << "\n"; */
                f.push_back(state.f.at(state.f.size() - (arity['f'] - i)));
            }
            
            for (int i = 0; i < arity['c']; i++) {
                /* cout << "push back float arg for " << program.at(i)->name << "\n"; */
                c.push_back(state.c.at(state.c.size() - (arity['c'] - i)));
            }
            
            for (int i = 0; i < arity['b']; i++) {
                /* cout << "push back bool arg for " << program.at(i)->name << "\n"; */
                b.push_back(state.b.at(state.b.size() - (arity['b'] - i)));
            }
        }
        
    #else
        using namespace Pop::Op;
         
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
                f.row(r) = (f.row(r) < MIN_FLT).select(MIN_FLT,f.row(r));
                f.row(r) = (f.row(r) > MAX_FLT).select(MAX_FLT,f.row(r));
                f.row(r) = (isnan(f.row(r))).select(0,f.row(r));
            }
            
            for (unsigned r = 0 ; r < c.rows(); ++r)
            {
                c.row(r) = (c.row(r) < MIN_FLT).select(MIN_FLT,c.row(r));
                c.row(r) = (c.row(r) > MAX_FLT).select(MAX_FLT,c.row(r));
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
        
        void State::copy_to_host(float* host_f, int increment)
        {
            copy_from_device((dev_f+increment), host_f, N); 
        }
        
        void State::copy_to_host(int* host_i, int increment)
        {
            copy_from_device((dev_c+increment), host_i, N); 
        }
        
        State::~State()
        {
            free_device(dev_f, dev_c, dev_b);
        }
        
        void Trace::copy_to_trace(State& state, std::map<char, unsigned int> &arity)
        {
            int increment;
            
            for (int i = 0; i < arity['f']; i++)
            {
                ArrayXf tmp(state.N);
                
                increment = (state.idx['f'] - (arity['f'] - i))*state.N;
                
                /*cout << "State index " << state.idx['f']
                     << " i = "<<i
                     << " arity['f'] = " << arity['f']
                     << " increment = " << increment<<endl;*/
                
                copy_from_device((state.dev_f+increment), tmp.data(), state.N); 
                
                f.push_back(tmp.cast<float>());
            }
            
            for (int i = 0; i < arity['c']; i++)
            {
                ArrayXi tmp(state.N);
                
                increment = (state.idx['c'] - (arity['c'] - i))*state.N;
                copy_from_device((state.dev_c+increment), tmp.data(), state.N); 
                
                c.push_back(tmp);
            }
            
            for (int i = 0; i < arity['b']; i++)
            {
                ArrayXb tmp(state.N);
                
                increment = (state.idx['b'] - (arity['b'] - i))*state.N;
                copy_from_device((state.dev_b+increment), tmp.data(), state.N); 
                
                b.push_back(tmp);
            }
        }

    #endif
    }
}

