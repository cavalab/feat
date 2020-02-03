/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "population.h"

namespace FT{   


    namespace Pop{
        
        int last; 

        Population::Population(){}
        
        Population::Population(int p)
        {
            individuals.resize(p); 
            locs.resize(2*p); 
            std::iota(locs.begin(),locs.end(),0);
            for (unsigned i = 0; i < individuals.size(); ++i)
            {
                individuals.at(i).set_id(locs.at(i));
                individuals.at(i).set_parents(vector<int>(1,-1));
           }
        }
        
        Population::~Population(){}
        
        /// update individual vector size 
        void Population::resize(int pop_size, bool resize_locs)
        {	
            individuals.resize(pop_size); 
            if (resize_locs)        // if this is an initial pop size, locs should be resized
            {
                locs.resize(2*pop_size); 
                std::iota(locs.begin(),locs.end(),0);
            }
        }
        
        /// returns population size
        int Population::size(){ return individuals.size(); }

        const Individual Population::operator [](size_t i) const {return individuals.at(i);}
        
        const Individual & Population::operator [](size_t i) {return individuals.at(i);}


        void Population::init(const Individual& starting_model, const Parameters& params,
                              bool random)
        {
            /*!
             *create random programs in the population, seeded by initial model weights 
             */
            individuals.at(0) = starting_model;
            individuals.at(0).loc = 0;

            #pragma omp parallel for
            for (unsigned i = 1; i< individuals.size(); ++i)
            {          
                // pick a dimensionality for this individual
                int dim = r.rnd_int(1,params.max_dim);      
                // pick depth from [params.min_depth, params.max_depth]
                /* unsigned init_max = std::min(params.max_depth, unsigned int(3)); */
                int depth;
                if (random)
                    depth = r.rnd_int(1, params.max_depth);
                else
                    /* depth =  r.rnd_int(1, std::min(params.max_depth,unsigned(3))); */
                    depth =  r.rnd_int(1, params.max_depth);
                // make a program for each individual
                char ot = r.random_choice(params.otypes);
                individuals.at(i).program.make_program(params.functions, 
                                                    params.terminals, 
                                                    depth,
                                                    params.term_weights,
                                                    params.op_weights, 
                                                    dim, 
                                                    ot, 
                                                    params.longitudinalMap, 
                                                    params.ttypes);
                
                /* std::cout << individuals.at(i).program_str() + " -> "; */
                /* std::cout << individuals.at(i).get_eqn() + "\n"; */
               
                // set location of individual and increment counter             
                individuals.at(i).loc = i;   
            }
            // define open locations
            update_open_loc(); 
        }
       
       void Population::update(vector<size_t> survivors)
       {

           /*!
            * cull population down to survivor indices.
            */
           vector<size_t> pop_idx(individuals.size());
           std::iota(pop_idx.begin(),pop_idx.end(),0);
           std::reverse(pop_idx.begin(),pop_idx.end());
           for (const auto& i : pop_idx)
               if (!in(survivors,i))
                   individuals.erase(individuals.begin()+i);                         
              
           //individuals.erase(std::remove_if(individuals.begin(), individuals.end(), 
           //                  [&survivors](const Individual& ind){ return !in(survivors,ind.loc);}),
           //                  individuals.end());

           // reset the open locations in F matrix 
           update_open_loc();
       
       }

       size_t Population::get_open_loc()
       {
           /*!
            * grabs an open location and removes it from the vector.
            */
           size_t loc = open_loc.back(); open_loc.pop_back();
           return loc;
       }

       void Population::update_open_loc()
       {
           /*!
            * updates open_loc to any locations in [0, 2*popsize-1] not in individuals.loc
            */
           vector<size_t> current_locs, new_open_locs;
          
           for (const auto& ind : individuals)  // get vector of current locations
               current_locs.push_back(ind.loc);

           for (const auto& i : locs)           // find open locations       
            if (!in(current_locs,i))
                   new_open_locs.push_back(i); 
           
            open_loc = new_open_locs;      // re-assign open locations             
            //std::cout << "updating open_loc to ";
            //for (auto o: open_loc) std::cout << o << " "; std::cout << "\n";
       }

       void Population::add(Individual& ind)
       {
           /*!
            * adds ind to individuals, giving it an open location and bookeeping.
            */

           ind.loc = get_open_loc();
           individuals.push_back(ind);
       }

       string Population::print_eqns(bool just_offspring, string sep)
       {
           string output = "";
           int start = 0;
           
           if (just_offspring)
               start = individuals.size()/2;

           for (unsigned int i=start; i< individuals.size(); ++i)
               output += individuals.at(i).get_eqn() + sep;
           
           return output;
       }

        vector<size_t> Population::sorted_front(unsigned rank=1)
        {
            /* Returns individuals on the Pareto front, sorted by increasign complexity. */
            vector<size_t> pf;
            for (unsigned int i =0; i<individuals.size(); ++i)
            {
                if (individuals.at(i).rank == rank)
                    pf.push_back(i);
            }
            std::sort(pf.begin(),pf.end(),SortComplexity(*this)); 
            auto it = std::unique(pf.begin(),pf.end(),SameFitComplexity(*this));
            pf.resize(std::distance(pf.begin(),it));
            return pf;
        }
        
    }
    
}
