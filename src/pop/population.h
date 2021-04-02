/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef POPULATION_H
#define POPULATION_H

//#include "node.h" // including node.h since definition of node is in the header
#include "individual.h"
using std::vector;
using std::string;
using Eigen::Map;

namespace FT{ 

/**
 * @namespace FT::Pop
 * @brief namespace containing population representations used in Feat
 */
namespace Pop{   
////////////////////////////////////////////////////////////////// Declarations
extern int last;
/*!
 * @class Population
 * @brief Defines a population of programs and functions for constructing them. 
 */
struct Population
{
    vector<Individual> individuals;     ///< individual programs

    Population(int p = 0);
    
    ~Population();
    
    /// initialize population of programs with a starting model and/or from file 
    void init(const Individual& starting_model, 
              const Parameters& params, 
              bool random = false,
              string filename=""
              );
    
    /// update individual vector size 
    void resize(int pop_size);
    
    /// reduce programs to the indices in survivors. 
    void update(vector<size_t> survivors);
    
    /// returns population size
    int size();

    /// adds a program to the population. 
    void add(Individual&);
    
    /// setting and getting from individuals vector
    const Individual operator [](size_t i) const;
    
    const Individual & operator [](size_t i);

    /// return population equations. 
    string print_eqns(bool just_offspring=false, string sep="\n");

    /// return complexity-sorted Pareto front indices. 
    vector<size_t> sorted_front(unsigned);
    
    /// Sort population in increasing complexity.
    struct SortComplexity
    {
        Population& pop;
        SortComplexity(Population& p): pop(p){}
        bool operator()(size_t i, size_t j)
        { 
            return pop.individuals[i].set_complexity() < pop.individuals[j].set_complexity();
        }
    };
    
    /// check for same fitness and complexity to filter uniqueness. 
    struct SameFitComplexity
    {
        Population & pop;
        SameFitComplexity(Population& p): pop(p){}
        bool operator()(size_t i, size_t j)
        {
            return (pop.individuals[i].fitness == pop.individuals[j].fitness &&
                   pop.individuals[i].set_complexity() == pop.individuals[j].set_complexity());
        }
    };

    // save serialized population
    void save(string filename);
    // load serialized population
    void load(string filename);

};        
//TODO
/* void from_json(const json& j, Population& p); */
/* void to_json(json& j, const Population& p); */
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Population, individuals);
}//Pop
}//FT    
#endif
