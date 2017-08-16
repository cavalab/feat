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

    class Variation 
    {
        // methods for crossover and mutation of programs.
        public:
            //constructor
            Variation();
            //destructor
            ~Variation();
            //method to handle variation of population
            void vary(Population& pop, vector<size_t> parents, const Parameters& params);
            
        private:
            // crossover 
            void crossover(Individual& mom, Individual& dad, Individual& child);

            // mutation
            void mutation(Individual& mom, Individual& child);
    };
}
#endif
