/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef RND_H
#define RND_H
//external includes
#include <iostream>
#include <random>
#include <limits>
#include <vector>
#include <omp.h>

using namespace std;
using std::swap;

namespace FT {
    
    ////////////////////////////////////////////////////////////////////////////////// Declarations
    
    /*!
     * @class Rnd
     * @brief Defines a multi-core random number generator and its operators.
     */
    // forward declaration of Node class
    class Node;

    class Rnd
    {
        public:
            
            Rnd();

            void set_seed(int seed);

            
            int rnd_int( int lowerLimit, int upperLimit );

            float rnd_flt(float min=0.0, float max=1.0);

            double rnd_dbl(double min=0.0, double max=1.0);
            
            double operator()(unsigned i);
            
            float operator()();

			template <class RandomAccessIterator>
			void shuffle (RandomAccessIterator first, RandomAccessIterator last);
            
            template<typename Iter>                                    
            Iter select_randomly(Iter start, Iter end);
           
            template<typename T>
            T random_choice(const vector<T>& v);
 
           
            template<typename T, typename D>
            T random_choice(const vector<T>& v, const vector<D>& w );
            
            float gasdev();
            
            ~Rnd();

        private:
            vector<std::mt19937> rg;
     
    };

    /////////////////////////////////////////////////////////////////////////////////// Definitions
    static Rnd r;   // random number generator     
}
#endif
