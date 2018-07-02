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
    ////////////////////////////////////////////////////////////////////////////////// Declarations
    extern Rnd r;
    extern int last;
    /*!
     * @class Population
     * @brief Defines a population of programs and functions for constructing them. 
     */
    struct Population
    {
        vector<Individual> individuals;     ///< individual programs
        vector<size_t> open_loc;            ///< unfilled matrix positions
        vector<size_t> locs;
        

        Population();
        
        Population(int p);
        
        ~Population();
        
        /// initialize population of programs. 
        void init(const Individual& starting_model, const Parameters& params, bool random = false);
        
        /// update individual vector size 
        void resize(int pop_size, bool resize_locs=false);
        
        /// reduce programs to the indices in survivors. 
        void update(vector<size_t> survivors);
        
        /// returns population size
        int size();

        /// returns an open location 
        size_t get_open_loc(); 
        
        /// updates open locations to reflect population.
        void update_open_loc();

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
                return pop.individuals[i].complexity() < pop.individuals[j].complexity();
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
                       pop.individuals[i].complexity() == pop.individuals[j].complexity());
            }
        };

    };
 
    bool is_valid_program(NodeVector& program, unsigned num_features, 
                          vector<string> longitudinalMap);
   
    void make_tree(NodeVector& program, 
                   const NodeVector& functions, 
                   const NodeVector& terminals, int max_d,  
                   const vector<double>& term_weights, char otype, const vector<char>& term_types);

    void make_program(NodeVector& program, 
                      const NodeVector& functions, 
                      const NodeVector& terminals, int max_d, 
                      const vector<double>& term_weights, int dim, char otype, 
                      vector<string> longitudinalMap, const vector<char>& term_types);
    
}//FT    
#endif
