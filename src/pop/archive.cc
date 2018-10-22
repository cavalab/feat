/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "archive.h"

namespace FT{

    namespace Pop{

        Archive::Archive() : selector(true) {}
        
        bool Archive::sortComplexity(const Individual& lhs, const Individual& rhs)
        {
            return lhs.c < rhs.c;
        }

        bool Archive::sameFitComplexity(const Individual& lhs, const Individual& rhs)
        {
            return (lhs.fitness == rhs.fitness &&
                   lhs.get_complexity() == rhs.get_complexity());
        }
        
        void Archive::init(Population& pop) 
        {
           auto tmp = pop.individuals;
           selector.fast_nds(tmp); 
           /* vector<size_t> front = this->sorted_front(); */
           for (const auto& t : tmp )
           {
               if (t.rank ==1){
                   archive.push_back(t);
                   archive[archive.size()-1].complexity();
               }
           } 
           cout << "intializing archive with " << archive.size() << " inds\n"; 

           std::sort(archive.begin(),archive.end(), &sortComplexity); 
        }

        void Archive::update(const Population& pop, const Parameters& params)
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
            {
                archive.push_back(tmp.at(i));
                archive[archive.size()-1].complexity();
            }
            std::sort(archive.begin(),archive.end(),&sortComplexity); 
            auto it = std::unique(archive.begin(),archive.end(), &sameFitComplexity);
            archive.resize(std::distance(archive.begin(),it));
        }
    }
}
