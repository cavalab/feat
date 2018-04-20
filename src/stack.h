/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef STACK_H
#define STACK_H
#ifdef USE_CUDA
    #include "node-cuda/stack_utils.h"
#endif
//#include "node/node.h"
//external includes
#define MAX_DBL std::numeric_limits<double>::max()
#define MIN_DBL std::numeric_limits<double>::min()

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
        Array<float, Dynamic, Dynamic, RowMajor> f;
        Array<bool, Dynamic, Dynamic, RowMajor>  b;
        Stack<std::pair<vector<ArrayXd>, vector<ArrayXd> > > z;
        Stack<string> fs;
        Stack<string> bs;
        Stack<string> zs;

        float * dev_f; 
        bool * dev_b; 
        std::map<char, size_t> idx; 
        size_t N; 

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
            /* std::cout << "before dev_allocate, dev_f is " << dev_f << "\n"; */
            dev_allocate(dev_f, dev_b, N*stack_size.at('f'), N*stack_size.at('b'));
            /* std::cout << "after dev_allocate, dev_f is " << dev_f << "\n"; */
            this->N = N;
            f.resize(stack_size.at('f'),N);
            b.resize(stack_size.at('b'),N);
        }

        void limit()
        {
            // clean floating point stack. 
            for (unsigned r = 0 ; r < f.rows(); ++r)
            {
                f.row(r) = (isinf(f.row(r))).select(MAX_DBL,f.row(r));
                f.row(r) = (isnan(f.row(r))).select(0,f.row(r));
            }
        }
        /// resize the f and b stacks to match the outputs of the program
        void trim()
        {
            /* std::cout << "resizing f to " << idx['f'] << "x" << f.cols() << "\n"; */
            f.resize(idx['f'],f.cols());
            b.resize(idx['b'],b.cols());
            /* std::cout << "new f size: " << f.size() << "," << f.rows() << "x" << f.cols() << "\n"; */
            /* usigned frows = f.rows()-1; */
            /* for (unsigned r = idx['f']; r < f.rows(); ++r) */
            /*     f.block(r,0,frows-r,f.cols()) = f.block(r+1,0,frows-r,f.cols()); */
            /*     f.conservativeResize(frows,f.cols()); */
        }
        void copy_to_host(const std::map<char, size_t>& stack_size)
        {
            /* std::cout << "size of f before copy_from_device: " << f.size() */ 
            /*           << ", stack size: " << N*stack_size.at('f') << "\n"; */
            /* std::cout << "size of b before copy_from_device: " << b.size() */ 
            /*           << ", stack size: " << N*stack_size.at('b') << "\n"; */

            copy_from_device(dev_f, f.data(), dev_b, b.data(), N*stack_size.at('f'), 
                             N*stack_size.at('b'));
            trim(); 
            limit();
        }
        ~Stacks()
        {
            free_device(dev_f, dev_b);
        }
    };
#endif
}



#endif
