/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef VARIATION_H
#define VARIATION_H

#include <iostream>
using namespace std;

#include "../pop/nodevector.h"
#include "../pop/population.h"
#include "../params.h"

namespace FT{

    /**
     * @namespace FT::Vary
     * @brief namespace containing various variation methods for cross and 
     * mutation in Feat
     */
    namespace Vary{

        //struct Individual;  // forward declarations
        //struct Parameters;
        //struct Population;
        //Rnd r;
        ////////////////////////////////////////////////////////// Declarations
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
                void vary(Population& pop, const vector<size_t>& parents, 
                        const Parameters& params, const Data& d);
                
                void delete_mutate(Individual& child, 
                        const Parameters& params);
                void delete_dimension_mutate(Individual& child, 
                        const Parameters& params);
                bool correlation_delete_mutate(Individual& child, 
                        MatrixXf Phi, const Parameters& params, const Data& d);
            private:
            
                /// crossover
                bool cross(const Individual& mom, const Individual& dad, 
                        Individual& child, const Parameters& params, 
                        const Data& d);
                
                /// residual crossover
                bool residual_cross(const Individual& mom, const Individual& dad, 
                        Individual& child, const Parameters& params, 
                        const Data& d);

                /// stagewise crossover 
                bool stagewise_cross(const Individual& mom, const Individual& dad, 
                        Individual& child, const Parameters& params, 
                        const Data& d);

                /// mutation
                bool mutate(const Individual& mom, Individual& child, 
                        const Parameters& params, const Data& d);
                void point_mutate(Individual& child, const Parameters& params);
                void insert_mutate(Individual& child, const Parameters& params);
     
                /// splice two programs together
                void splice_programs(NodeVector& vnew, 
                                     const NodeVector& v1, size_t i1, size_t j1, 
                                     const NodeVector& v2, size_t i2, size_t j2);
                /// debugging printout of crossover operation.
                void print_cross(const Individual&,size_t,size_t,
                        const Individual&, size_t, size_t, Individual&, 
                        bool after=true);       
                
                float cross_rate;     ///< fraction of crossover in total variation
        };

        std::unique_ptr<Node> random_node(const NodeVector & v);
    }
}
#endif
