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
            return lhs.c < rhs.c;
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
                   archive.push_back(t);
                   archive.at(archive.size()-1).complexity();
               }
           } 
           cout << "intializing archive with " << archive.size() << " inds\n"; 
           if (this->sort_complexity)
               std::sort(archive.begin(),archive.end(), &sortComplexity); 
           else
               std::sort(archive.begin(),archive.end(), &sortObj1); 

        }

        void Archive::update(const Population& pop, const Parameters& params)
        {
                        
            vector<Individual> tmp = pop.individuals;

            #pragma omp parallel for
            for (unsigned int i=0; i<tmp.size(); ++i)
                tmp.at(i).set_obj(params.objectives);

            for (const auto& p : archive)
                tmp.push_back(p);

            selector.fast_nds(tmp);
            
            vector<int> pf = selector.front.at(0);
          
            archive.resize(0);  // clear archive
            for (const auto& i : pf)   // refill archive with new pareto front
            {
                archive.push_back(tmp.at(i));
                archive.at(archive.size()-1).complexity();
            }
            if (this->sort_complexity)
                std::sort(archive.begin(),archive.end(),&sortComplexity); 
            else
                std::sort(archive.begin(),archive.end(), &sortObj1); 
            /* auto it = std::unique(archive.begin(),archive.end(), &sameFitComplexity); */
            auto it = std::unique(archive.begin(),archive.end(), 
                    &sameObjectives);
            archive.resize(std::distance(archive.begin(),it));
        }
    }
}
