/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef PARAMS_H
#define PARAMS_H

namespace FT{

    struct Parameters
    {
        // holds the hyperparameters for Fewtwo. 
        int pop_size;           // population size
        int gens;               // max generations
        string ml;              // machine learner with which Fewtwo is paired
        bool classification;    // flag to conduct classification rather than regression
        int max_stall;          // maximum stall in learning (termination criterion) in generations
        char otype;             // output type of the programs ('f': float, 'b': boolean)
        int verbosity;          // amount of printing. 0: none, 1: minimal, 2: all

        Parameters(int pop_size, int gens, string& ml, bool classification, int max_stall, 
                   char otype, int vebosity):    
            pop_size(pop_size),
            gens(gens),
            ml(ml),
            classification(classification),
            max_stall(max_stall),
            otype(otype), 
            verbosity(verbosity){}
        
        ~Parameters(){}
        
        void msg(string m, int v)
        {
            /* prints messages based on verbosity level. */

            if (verbosity >= v)
                std::cout << m << "...\n";
        }
    };
}
#endif

