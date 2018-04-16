/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef STACK_H
#define STACK_H
#ifdef USE_CUDA
    #include "node-cuda/cuda_utils.h"
#endif
//#include "node/node.h"
//external includes

namespace FT
{
    template<typename type>
    class Stack
    {
        private:
            std::vector<type> st;
            
        public:
        
            Stack()
            {
                st = std::vector<type>();
            }
            
            void push(type element){ st.push_back(element); }
            
            type pop()
            {
                type ret = st.back();
                st.pop_back();
                return ret;
            }
            
            bool empty(){ return st.empty(); }
            
            unsigned int size(){ return st.size(); }
            
            type& top(){ return st.back(); }
            
            type& at(int i){ return st.at(i); }
            
            void clear(){ st.clear(); }
            
            typename vector<type>::iterator begin(){ return st.begin(); }
            
            typename vector<type>::iterator end(){ return st.end(); }
            
            typename vector<type>::const_iterator begin() const { return st.begin(); }
            
            typename vector<type>::const_iterator end() const { return st.end(); }
    };
#ifndef USE_CUDA    
    struct Stacks
    {
        Stack<ArrayXd> f;
        Stack<ArrayXb> b;
        Stack<std::pair<vector<ArrayXd>, vector<ArrayXd> > > z;
        Stack<string> fs;
        Stack<string> bs;
        Stack<string> zs;
        
        bool check(std::map<char, unsigned int> &arity)
        {
            if(arity.find('z') == arity.end())
                return (f.size() >= arity['f'] && b.size() >= arity['b']);
            else
                return (f.size() >= arity['f'] && b.size() >= arity['b'] 
                        && z.size() >= arity['z']);
        }
        
        bool check_s(std::map<char, unsigned int> &arity)
        {
            if(arity.find('z') == arity.end())
                return (fs.size() >= arity['f'] && bs.size() >= arity['b']);
            else
                return (fs.size() >= arity['f'] && bs.size() >= arity['b'] 
                        && zs.size() >= arity['z']);
        }
    };
#else
    struct Stacks
    {
        ArrayXXf f;
        ArrayXXb b;
        Stack<std::pair<vector<ArrayXd>, vector<ArrayXd> > > z;
        Stack<string> fs;
        Stack<string> bs;
        Stack<string> zs;

        float * dev_f; 
        bool * dev_b; 
        std::map<char, size_t> idx; 
        int N; 

        Stacks()
        {
            idx['f']=0;
            idx['b']=0;
        }

        void update_idx(char otype, std::map<char, unsigned>& arity)
        {
            ++idx[otype];
            for (const auto& a : arity)
                    idx[a.first] -= a.second;
        }
        bool check(std::map<char, unsigned int> &arity)
        {
            if(arity.find('z') == arity.end())
                return (f.rows() >= arity['f'] && b.rows() >= arity['b']);
            else
                return (f.rows() >= arity['f'] && b.rows() >= arity['b'] 
                        && z.size() >= arity['z']);
        }
        
        bool check_s(std::map<char, unsigned int> &arity)
        {
            if(arity.find('z') == arity.end())
                return (fs.size() >= arity['f'] && bs.size() >= arity['b']);
            else
                return (fs.size() >= arity['f'] && bs.size() >= arity['b'] 
                        && zs.size() >= arity['z']);
        }
        
        void allocate(const std::map<char, size_t>& stack_size, size_t N)
        {
            allocate(dev_f, dev_b, N*stack_size['f'], N*stack_size['b']);
            N = N;
        }

        void copy_from_device(const std::map<char, size_t>& stack_size)
        {
            copy_from_device(dev_f, f.data(), dev_b, b.data(), N*stack_size['f'], N*stack_size['b']);
        }
        ~Stacks()
        {
            // Free memory
            cudaFree(dev_f); 
            cudaFree(dev_b);         
        }
    };
#endif
}



#endif
