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
        float cross_ratio;      // fraction of crossover for variation
        int max_stall;          // maximum stall in learning (termination criterion) in generations
        char otype; 

        Parameters(int pop_size, int gens, string& ml, bool classification, float cross_ratio, 
                   int max_stall, char otype): pop_size(pop_size),
                                   gens(gens),
                                   ml(ml),
                                   classification(classification),
                                   cross_ratio(cross_ratio),
                                   max_stall(max_stall),
                                   otype(otype)
        {};
        ~Parameters();
    };
}
#endif

