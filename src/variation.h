/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef VARIATION_H
#define VARIATION_H

namespace FT{
    struct Individual;  // forward declarations
    struct Parameters;
    struct Population;

    ////////////////////////////////////////////////////////////////////////////////// Declarations
    
    class Variation 
    {
        /* methods for crossover and mutation of programs. */

        public:
            //constructor
            Variation(float cr){cross_ratio=cr;}
            
            //destructor
            ~Variation(){}

            //method to handle variation of population
            void vary(Population& pop, const vector<size_t> parents, const Parameters& params);
            
        private:
            // crossover 
            void cross(const Individual& mom, const Individual& dad, Individual& child);

            // mutation
            void mutate(const Individual& mom, Individual& child);
            
            float cross_ratio;     // ratio of crossover in total variation
    };


    /////////////////////////////////////////////////////////////////////////////////// Definitions
    
    void Variation::vary(Population& pop, const vector<size_t> parents, const Parameters& params)
    {
        /* performs variation on the current population. 
         *
         * Input:
         *      pop: current population
         *      parents: indices of population to use for variation
         *      params: fewtwo parameters
         * Output:
         *      pop: appends params.pop_size offspring derived from parent variation
         */
        
        Individual child;

        for (auto ind : pop.individuals)
        {
            float rand;
            if ( rand < cross_ratio)    // crossover
            {
                // get random mom and dad 
                // create child
                
            }
            else                // mutation
            {
                // get random mom 
                // create child
                
            }
            //push child into pop
        }
    }

    void Variation::mutate(const Individual& mom, Individual& child)
    {
        /* 1/n point mutation
         * 
         * Input:
         *      mom: root parent
         * Output:
         *      child: copy of mom with some mutations
         */

    }

    void Variation::cross(const Individual& mom, const Individual& dad, Individual& child)
    {
        /* subtree crossover
         *
         * Input:
         *      mom: root parent
         *      dad: parent from which subtree is chosen
         * Output:
         *      child: mom with dad subtree graft
         */

    }

}
#endif
