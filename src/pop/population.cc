/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "population.h"

namespace FT{   
namespace Pop{
        
int last; 

Population::Population(int p)
{
    individuals.resize(p); 
    for (unsigned i = 0; i < individuals.size(); ++i)
    {
        individuals.at(i).set_parents(vector<int>(1,-1));
   }
}

Population::~Population(){}

/// update individual vector size 
void Population::resize(int pop_size)
{	
    individuals.resize(pop_size); 
}

/// returns population size
int Population::size(){ return individuals.size(); }

const Individual Population::operator [](size_t i) const {return individuals.at(i);}

const Individual & Population::operator [](size_t i) {return individuals.at(i);}


void Population::init(const Individual& starting_model, 
                      const Parameters& params,
                      bool random,
                      string filename
                      )
{
    /*!
     *create random programs in the population, 
     * seeded by initial model weights and potentially saved models
     */
    int starting_size = 1;

    if (!filename.empty())
    {
        this->load(filename);

        starting_size = this->individuals.size();
        cout << "individuals.size(): " << this->individuals.size() << endl;
        cout << "params.pop_size: " << params.pop_size << endl;

        if (starting_size < params.pop_size)
        {
            individuals.resize(params.pop_size);
            individuals.at(starting_size) = starting_model;
            ++starting_size;
        }
        else if (starting_size == params.pop_size)
        {
            WARN("The first model provided in the startup file is going "
                 "to get overwritten by the initial (linear) model. "
                 "To avoid this, increase pop_size by 1."
                 ); 
            individuals.at(0) = starting_model;
        }
        else
        {
            WARN("Too many models provided in the startup file ("
                 + to_string(starting_size) + "); pop_size is set to "
                 + to_string(params.pop_size) + ". Also, the first model is going "
                 "to get overwritten by the initial (linear) model."); 
            individuals.at(0) = starting_model;
        }
    }
    else
        individuals.at(0) = starting_model;


    /* if the starting pop is smaller than the 
     * set size of the population, initialize
     * additional individuals.*/
    #pragma omp parallel for
    for (unsigned i = starting_size; i< individuals.size(); ++i)
    {          
        individuals.at(i).initialize(params, random, i);
    }
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
      
}

void Population::add(Individual& ind)
{
   /*!
    * adds ind to individuals, giving it an open location and bookeeping.
    */

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
        
void Population::save(string filename)
{
    std::ofstream out;                      
    if (!filename.empty())
        out.open(filename);
    else
        out.open("pop.json");

    json j;
    to_json(j, *this);
    out << j ;
    out.close();
    logger.log("Saved population to file " + filename, 1);
}

void Population::load(string filename)
{
    //TODO: replace with from_json(j, this) call
    std::ifstream indata;
    indata.open(filename);
    if (!indata.good())
        THROW_INVALID_ARGUMENT("Invalid input file " + filename + "\n"); 

    std::string line;
    indata >> line; 

    json j = json::parse(line);
    from_json(j, *this);

    logger.log("Loaded population from " + filename + " of size = " 
            + to_string(this->size()),1);

    indata.close();
}


} // Pop
} // FT
