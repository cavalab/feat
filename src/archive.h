/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef ARCHIVE_H
#define ARCHIVE_H

//#include "node.h" // including node.h since definition of node is in the header
#include "individual.h"
#include "selection/nsga2.h"
using std::vector;
using std::string;
using Eigen::Map;

namespace FT{    
    ////////////////////////////////////////////////////////////////////////////////// Declarations
    /*!
     * @class Archive 
     * @brief Defines a Pareto archive of programs.
     */
    struct Archive  
    {
        vector<Individual> archive;         ///< individual programs in the archive

        NSGA2 selector;                     ///< nsga2 selection operator used for getting the front

        Archive() : selector(true) {}

        /// Sort population in increasing complexity.
        static bool sortComplexity(Individual& lhs, Individual& rhs)
        {
            return lhs.complexity() < rhs.complexity();
        }

        static bool sameFitComplexity(Individual& lhs, Individual& rhs)
        {
            return (lhs.fitness == rhs.fitness &&
                   lhs.complexity() == rhs.complexity());
        }

        void init(Population& pop) 
        {
           auto tmp = pop.individuals;
           selector.fast_nds(tmp); 
           /* vector<size_t> front = this->sorted_front(); */
           for (const auto& t : tmp )
           {
               if (t.rank ==1) archive.push_back(t);
           } 
           cout << "intializing archive with " << archive.size() << " inds\n"; 

           std::sort(archive.begin(),archive.end(), &sortComplexity); 
        }

        void update(const Population& pop, const Parameters& params)
        {
                        
            vector<Individual> tmp = pop.individuals;

            #pragma omp parallel for
            for (unsigned int i=0; i<tmp.size(); ++i)
                tmp[i].set_obj(params.objectives);

            for (const auto& p : archive)
                tmp.push_back(p);
 
            selector.fast_nds(tmp);
            
            vector<int> pf = selector.front[0];
          
            archive.resize(0);  // clear archive
            for (const auto& i : pf)   // refill archive with new pareto front
                archive.push_back(tmp.at(i));
             
            std::sort(archive.begin(),archive.end(),&sortComplexity); 
            auto it = std::unique(archive.begin(),archive.end(), &sameFitComplexity);
            archive.resize(std::distance(archive.begin(),it));
        }
       
    };
}
#endif
