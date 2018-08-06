/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "stack.h"

#include "node/node.h"
namespace FT
{

#ifndef USE_CUDA

    bool Stacks::check(std::map<char, unsigned int> &arity)
    {
        if(arity.find('z') == arity.end())
            return (f.size() >= arity['f'] && b.size() >= arity['b']);
        else
            return (f.size() >= arity['f'] && b.size() >= arity['b'] 
                    && z.size() >= arity['z']);
    }
    
    ///< checks if arity of node provided satisfies the node names in various string stacks
    bool Stacks::check_s(std::map<char, unsigned int> &arity)
    {
        if(arity.find('z') == arity.end())
            return (fs.size() >= arity['f'] && bs.size() >= arity['b']);
        else
            return (fs.size() >= arity['f'] && bs.size() >= arity['b'] 
                    && zs.size() >= arity['z']);
    }

#else
    Stacks::Stacks()
    {
        idx['f']=0;
        idx['b']=0;
    }
    
    void Stacks::update_idx(char otype, std::map<char, unsigned>& arity)
    {
        ++idx[otype];
        for (const auto& a : arity)
                idx[a.first] -= a.second;
    }
    
    bool Stacks::check(std::map<char, unsigned int> &arity)
    {
        if(arity.find('z') == arity.end())
            return (f.rows() >= arity['f'] && b.rows() >= arity['b']);
        else
            return (f.rows() >= arity['f'] && b.rows() >= arity['b'] 
                    && z.size() >= arity['z']);
    }
    
    bool Stacks::check_s(std::map<char, unsigned int> &arity)
    {
        if(arity.find('z') == arity.end())
            return (fs.size() >= arity['f'] && bs.size() >= arity['b']);
        else
            return (fs.size() >= arity['f'] && bs.size() >= arity['b'] 
                    && zs.size() >= arity['z']);
    }
    
    void Stacks::allocate(const std::map<char, size_t>& stack_size, size_t N)
    {
        //std::cout << "before dev_allocate, dev_f is " << dev_f << "\n";
        dev_allocate(dev_f, dev_b, N*stack_size.at('f'), N*stack_size.at('b'));
        //std::cout << "after dev_allocate, dev_f is " << dev_f << "\n";

	//printf("Allocated Stack Sizes\n");
	//printf("\tFloating stack N=%zu and stack size as %zu\n",N, stack_size.at('f'));

        this->N = N;
	
        f.resize(stack_size.at('f'),N);
        b.resize(stack_size.at('b'),N);
    }

    void Stacks::limit()
    {
        // clean floating point stack. 
        for (unsigned r = 0 ; r < f.rows(); ++r)
        {
            f.row(r) = (isinf(f.row(r))).select(MAX_DBL,f.row(r));
            f.row(r) = (isnan(f.row(r))).select(0,f.row(r));
        }
    }
    
    /// resize the f and b stacks to match the outputs of the program
    void Stacks::trim()
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
	b.conservativeResize(idx['b'], b.cols());
    }
    
    void Stacks::copy_to_host(const std::map<char, size_t>& stack_size)
    {
        /* std::cout << "size of f before copy_from_device: " << f.size() */ 
        /*           << ", stack size: " << N*stack_size.at('f') << "\n"; */
        /* std::cout << "size of b before copy_from_device: " << b.size() */ 
        /*           << ", stack size: " << N*stack_size.at('b') << "\n"; */
	 
        copy_from_device(dev_f, f.data(), dev_b, b.data(), N*idx['f'], N*idx['b']);
        //copy_from_device(dev_f, f.data(), dev_b, b.data(), N*stack_size.at('f'), N*stack_size.at('b'));

        trim(); 
        limit();
    }
    
    Stacks::~Stacks()
    {
	//printf("Calling free device\n");
        free_device(dev_f, dev_b);
	//cudaDeviceReset();
	//printf("Device freed\n");
    }

#endif    
    template class Stack<ArrayXd>;
    template class Stack<ArrayXb>;
    template class Stack<std::pair<vector<ArrayXd>, vector<ArrayXd> > >;
    template class Stack<string>;
    
}

