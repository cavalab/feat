/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "archive.h"

namespace FT{

    namespace Pop{

        Archive::Archive():  selector(true) {};

        void Archive::set_objectives(vector<string> objectives)
        {

            this->sort_complexity = in(objectives,std::string("complexity"));
        }


        
        bool Archive::sortComplexity(const Individual& lhs, 
                const Individual& rhs)
        {
            return lhs.complexity < rhs.complexity;
        }
        
        bool Archive::sortObj1(const Individual& lhs, 
                const Individual& rhs)
        {
            return lhs.obj.at(0) < rhs.obj.at(0);
        }

        bool Archive::sameFitComplexity(const Individual& lhs, 
                const Individual& rhs)
        {
            return (lhs.fitness == rhs.fitness &&
                   lhs.get_complexity() == rhs.get_complexity());
        }

        bool Archive::sameObjectives(const Individual& lhs, 
                const Individual& rhs)
        {
            for (const auto& o_lhs : lhs.obj)
            {
                for (const auto& o_rhs : rhs.obj)
                {
                    if (o_lhs != o_rhs)
                        return false;
                    
                }
            }
            return true;
        }
        
        void Archive::init(Population& pop) 
        {
           auto tmp = pop.individuals;
           selector.fast_nds(tmp); 
           /* vector<size_t> front = this->sorted_front(); */
           for (const auto& t : tmp )
           {
               if (t.rank ==1){
                   individuals.push_back(t);
                   individuals.at(individuals.size()-1).set_complexity();
               }
           } 
           cout << "intializing archive with " << individuals.size() << " inds\n"; 
           if (this->sort_complexity)
               std::sort(individuals.begin(),individuals.end(), &sortComplexity); 
           else
               std::sort(individuals.begin(),individuals.end(), &sortObj1); 

        }

        void Archive::update(const Population& pop, const Parameters& params)
        {
                        
            vector<Individual> tmp = pop.individuals;

            #pragma omp parallel for
            for (unsigned int i=0; i<tmp.size(); ++i)
                tmp.at(i).set_obj(params.objectives);

            for (const auto& p : individuals)
                tmp.push_back(p);

            selector.fast_nds(tmp);
            
            vector<int> pf = selector.front.at(0);
          
            individuals.resize(0);  // clear archive
            for (const auto& i : pf)   // refill archive with new pareto front
            {
                individuals.push_back(tmp.at(i));
                individuals.at(individuals.size()-1).set_complexity();
            }
            if (this->sort_complexity)
                std::sort(individuals.begin(),individuals.end(),&sortComplexity); 
            else
                std::sort(individuals.begin(),individuals.end(), &sortObj1); 
            /* auto it = std::unique(individuals.begin(),individuals.end(), &sameFitComplexity); */
            auto it = std::unique(individuals.begin(),individuals.end(), 
                    &sameObjectives);
            individuals.resize(std::distance(individuals.begin(),it));
        }
    }
}
