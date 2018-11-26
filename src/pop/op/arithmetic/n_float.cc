/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_float.h"
    	
namespace FT{

    namespace Pop{
        namespace Op{
            template <>
            NodeFloat<bool>::NodeFloat()
            {
                name = "b2f";
                otype = 'f';
                arity['b'] = 1;
                complexity = 1;
            }
            
            template <>
            NodeFloat<int>::NodeFloat()
            {
                name = "c2f";
                otype = 'f';
                arity['c'] = 1;
                complexity = 1;
            }

            #ifndef USE_CUDA 
            /// Evaluates the node and updates the state states. 
            template <class T>
            void NodeFloat<T>::evaluate(const Data& data, State& state)
            {
                state.push<float>(state.pop<T>().template cast<float>());
            }
            #else
            template <class T>
            void NodeFloat<T>::evaluate(const Data& data, State& state)
            {
                if(arity['b'])
                    GPU_Float(state.dev_f, state.dev_b, state.idx[otype], state.idx['b'], state.N);
                else
                    GPU_Float(state.dev_f, state.dev_c, state.idx[otype], state.idx['c'], state.N);
            }
            #endif

            /// Evaluates the node symbolically
            template <class T>
            void NodeFloat<T>::eval_eqn(State& state)
            {
                state.push<float>("float(" + state.popStr<T>() + ")");
            }
            
            template <class T>
            NodeFloat<T>* NodeFloat<T>::clone_impl() const { return new NodeFloat<T>(*this); }

            template <class T>
            NodeFloat<T>* NodeFloat<T>::rnd_clone_impl() const { return new NodeFloat<T>(*this); }  
        }
    }
}
