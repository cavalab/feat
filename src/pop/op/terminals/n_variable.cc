/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_variable.h"
			
namespace FT{

    namespace Pop{
        namespace Op{
        
            template <class T>
            NodeVariable<T>::NodeVariable()
            {
                name = "variable";
                loc = -1;
                complexity = 1;
            }

            template <class T>
            NodeVariable<T>::NodeVariable(const size_t& l, char ntype, std::string n)
            {
                name = "variable";
                if (n.empty())
	                variable_name = "x_" + std::to_string(l);
                else
                    variable_name = n;
	            otype = ntype;
	            complexity = 1;
	            loc = l;
            }

            #ifndef USE_CUDA
            /// Evaluates the node and updates the state states. 
            template <class T>		
            void NodeVariable<T>::evaluate(const Data& data, State& state)
            {
                state.push<T>(data.X.row(loc).template cast<T>());
            }
            
            #else
            template <class T>
            void NodeVariable<T>::evaluate(const Data& data, State& state)
            {
                if(otype == 'b')
                {
                    ArrayXb tmp = data.X.row(loc).cast<bool>();
                    GPU_Variable(state.dev_b, tmp.data(), state.idx[otype], state.N);
                }
                else if (otype == 'c')
                {
                    ArrayXi tmp = data.X.row(loc).cast<int>();
                    GPU_Variable(state.dev_c, tmp.data(), state.idx[otype], state.N);
                }
                else
                {
                    ArrayXf tmp = data.X.row(loc).cast<float>() ;
                    GPU_Variable(state.dev_f, tmp.data(), state.idx[otype], state.N);
                }
            }
            #endif

            /// Evaluates the node symbolically
            template <class T>
            void NodeVariable<T>::eval_eqn(State& state)
            {
                state.push<T>(variable_name);
            }

            template <class T>
            NodeVariable<T>* NodeVariable<T>::clone_impl() const { return new NodeVariable<T>(*this); }
              
            // rnd_clone is just clone_impl() for variable, since rand vars not supported
            template <class T>
            NodeVariable<T>* NodeVariable<T>::rnd_clone_impl() const { return new NodeVariable<T>(*this); }
            
            template class NodeVariable<bool>;
            template class NodeVariable<int>;
            template class NodeVariable<float>;
        }
    }
}
