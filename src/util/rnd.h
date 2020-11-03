/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef RND_H
#define RND_H
//external includes
#include <random>
#include <limits>
#include <vector>

#include "../init.h"
#include "error.h"

using namespace std;
using std::swap;

namespace FT {

    namespace Util{
    
        ////////////////////////////////////////////////////////////////////////////////// Declarations
        
        /*!
         * @class Rnd
         * @brief Defines a multi-core random number generator and its operators.
         */

        class Rnd
        {
            public:
                
                static Rnd* initRand();
                
                static void destroy();

                void set_seed(int new_seed);
        
                int get_seed(){return this->seed;};
                
                int rnd_int( int lowerLimit, int upperLimit );

                float rnd_flt(float min=0.0, float max=1.0);

                float rnd_dbl(float min=0.0, float max=1.0);
                
                float operator()(unsigned i);
                
                float operator()();

			    template <class RandomAccessIterator>
			    void shuffle (RandomAccessIterator first, 
                        RandomAccessIterator last)
			    {
	                for (auto i=(last-first)-1; i>0; --i) 
                    {
	                    std::uniform_int_distribution<decltype(i)> d(0,i);
		                swap (first[i], first[d(rg[omp_get_thread_num()])]);
	                }
	            }    
                
                template<typename Iter>                                    
                Iter select_randomly(Iter start, Iter end)
                {
                    std::uniform_int_distribution<> dis(0, 
                            distance(start, end) - 1);
                    advance(start, dis(rg[omp_get_thread_num()]));
                    return start;
                }
               
                template<typename T>
                T random_choice(const vector<T>& v)
                {
                   /*!
                    * return a random element of a vector.
                    */          
                    assert(v.size()>0 && 
                       " attemping to return random choice from empty vector");
                    return *select_randomly(v.begin(),v.end());
                }
     
               
                template<typename T, typename D>
                T random_choice(const vector<T>& v, const vector<D>& w )
                {
                    /*!
                     * return a weighted random element of a vector
                     */
                     
                    if(w.size() == 0)
                    {   
                        if (v.size() == 0)
                            THROW_LENGTH_ERROR("random_choice() called with "
                                    "w.size() = 0 and v.size() = 0");
                        else
                        {
                            THROW_LENGTH_ERROR("w.size() = 0, v.size() = "
                                    +to_string(v.size())+
                                    "; Calling random_choice(v)");
                            
                            return random_choice(v);
                        }
                    }
                    if(w.size() != v.size())
                    {   
                        cout<<"WARN! random_choice() w.size() " << w.size() << "!= v.size() " 
                            << v.size() << ", Calling random_choice(v)\n";
                        return random_choice(v);
                    }
                    else
                    {
                        assert(v.size() == w.size());
                        std::discrete_distribution<size_t> dis(w.begin(), w.end());
                        return v[dis(rg[omp_get_thread_num()])]; 
                    }
                }
                
                float gasdev();

            private:

                Rnd();
            
                ~Rnd();
                
                vector<std::mt19937> rg;
                
                static Rnd* instance;

                int seed;
         
        };
        
        static Rnd &r = *Rnd::initRand();
    }
}
#endif
