/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef ARCHIVE_H
#define ARCHIVE_H

//#include "node.h" // including node.h since definition of node is in the header
#include "individual.h"
#include "../sel/nsga2.h"
using std::vector;
using std::string;
using Eigen::Map;

namespace FT{

using namespace Sel;
///////////////////////////////////////////////////////////////////Declarations
/*!
 * @class Archive 
 * @brief Defines a Pareto archive of programs.
 */
     
namespace Pop{
    
struct Archive  
{
    vector<Individual> individuals; ///< individual programs in the archive
    bool sort_complexity;    ///< whether to sort archive by complexity

    NSGA2 selector; ///< nsga2 selection operator for getting the front

    Archive();
    void set_objectives(vector<string> objectives);
    /// Sort population in increasing complexity.
    static bool sortComplexity(const Individual& lhs, 
            const Individual& rhs);
    /// Sort population by first objective.
    static bool sortObj1(const Individual& lhs, 
            const Individual& rhs);
    /// check for repeats
    static bool sameFitComplexity(const Individual& lhs, 
            const Individual& rhs);
    static bool sameObjectives(const Individual& lhs, 
            const Individual& rhs);

    void init(Population& pop);

    void update(const Population& pop, const Parameters& params);
   
};
//serialization
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Archive, individuals);
} // Pop
} // FT
#endif
