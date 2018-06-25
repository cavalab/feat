/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef VARIATION_H
#define VARIATION_H

#include <iostream>
using namespace std;

#include "nodevector.h"
#include "population.h"
#include "params.h"

namespace FT{
    //struct Individual;  // forward declarations
    //struct Parameters;
    //struct Population;
    //Rnd r;
    ////////////////////////////////////////////////////////////////////////////////// Declarations
    /*!
     * @class Variation
     */ 
    class Variation 
    {
        /*!
         * methods for crossover and mutation of programs.
         */

        public:
        
            /// constructor
            Variation(float cr);
                       
            /// update cross rate
            void set_cross_rate(float cr);
            
            /// return current cross rate
            float get_cross_rate();
            
             /// destructor
            ~Variation();

            /// method to handle variation of population
            void vary(Population& pop, const vector<size_t>& parents, const Parameters& params);
            
        private:
        
            /// crossover
            bool cross(Individual& mom, Individual& dad, Individual& child,
                       const Parameters& params);
            
            /// mutation
            bool mutate(Individual& mom, Individual& child, const Parameters& params);
            void point_mutate(Individual& child, const Parameters& params);
            void insert_mutate(Individual& child, const Parameters& params);
            void delete_mutate(Individual& child, const Parameters& params);
 
            /// splice two programs together
            void splice_programs(NodeVector& vnew, 
                                 const NodeVector& v1, size_t i1, size_t j1, 
                                 const NodeVector& v2, size_t i2, size_t j2);
            /// debugging printout of crossover operation.
            void print_cross(Individual&,size_t,size_t,Individual&, size_t, size_t, Individual&,
                             bool after=true);       
            
            float cross_rate;     ///< fraction of crossover in total variation
    };

    std::unique_ptr<Node> random_node(const NodeVector & v);
}
#endif
