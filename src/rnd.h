/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef RND_H
#define RND_H
//external includes
#include <random>
#include <limits>
using std::swap;

namespace FT {
    
    ////////////////////////////////////////////////////////////////////////////////// Declarations
    
    /*!
     * @class Rnd
     * @brief Defines a multi-core random number generator and its operators.
     */
    class Rnd
    {
        public:
            
            Rnd()
            {
                /*!
                 * need a random generator for each core to do multiprocessing
                 */
                rg.resize(omp_get_max_threads());                      
            }

            void set_seed(int seed)
            { 
                /*!
                 * set seeds for each core's random number generator
                 */
                if (seed == 0)
                {
                    std::random_device rd; 

                    for (auto& r : rg)
                        r.seed(rd());
                }
                else    // seed first rg with seed, then seed rest with random ints from rg[0]. 
                {
                    rg[0].seed(seed);
                    
                    int imax = std::numeric_limits<int>::max();
                    
                    std::uniform_int_distribution<> dist(0, imax);

                    for (size_t i = 1; i < rg.size(); ++i)
                        rg[i].seed(dist(rg[0]));                     
                }
            }


            int rnd_int( int lowerLimit, int upperLimit ) 
            {
                std::uniform_int_distribution<> dist( lowerLimit, upperLimit );
                return dist(rg[omp_get_thread_num()]);
            }

            float rnd_flt(float min=0.0, float max=1.0)
            {
                std::uniform_real_distribution<float> dist(min, max);
                return dist(rg[omp_get_thread_num()]);
            }

            double rnd_dbl(double min=0.0, double max=1.0)
            {
                std::uniform_real_distribution<double> dist(min, max);
                return dist(rg[omp_get_thread_num()]);
            }
            
            double operator()(unsigned i) 
            {
                return rnd_dbl(0.0,i);
            }
            
            float operator()() { return rnd_flt(0.0,1.0); }

			template <class RandomAccessIterator>
			void shuffle (RandomAccessIterator first, RandomAccessIterator last)
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
                std::uniform_int_distribution<> dis(0, distance(start, end) - 1);
                advance(start, dis(rg[omp_get_thread_num()]));
                return start;
            } 
            
            template<typename T>
            T random_choice(const vector<T>& v)
            {
               /*!
                * return a random element of a vector.
                */          
                assert(v.size()>0 && " attemping to return random choice from empty vector");
                return *select_randomly(v.begin(),v.end());
            }
            
            template<typename T, typename D>
            T random_choice(const vector<T>& v, const vector<D>& w )
            {
                /*!
                 * return a weighted random element of a vector
                 */
                assert(v.size() == w.size());
                    
                std::discrete_distribution<size_t> dis(w.begin(), w.end());

                return v[dis(rg[omp_get_thread_num()])]; 
            }
            ~Rnd() {}

    private:
        vector<std::mt19937> rg;
     
    };

    /////////////////////////////////////////////////////////////////////////////////// Definitions
    static Rnd r;   // random number generator     
}
#endif
