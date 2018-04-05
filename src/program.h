/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef PROGRAM_H
#define PROGRAM_H

namespace FT{
    
    ////////////////////////////////////////////////////////////////////////////////// Declarations

    /*!
     * @class Program
     * @brief an extension of a vector of unique pointers to nodes 
     */
    struct Program : public std::vector<std::unique_ptr<Node>> {
        
        Program() = default;
        ~Program() = default; 
        Program(const Program& other)
        {
            std::cout<<"in Program(const Program& other)\n";
            for (const auto& p : other)
                this->push_back(p->clone());
        }
        Program(Program && other) = default;
        /* { */
        /*     std::cout<<"in Program(Program&& other)\n"; */
        /*     for (const auto& p : other) */
        /*         this->push_back(p->clone()); */
        /* } */
        Program& operator=(Program const& other)
        { 

            std::cout << "in Program& operator=(Program const& other)\n";
            for (const auto& p : other)
                this->push_back(p->clone());
            return *this; 
        }        
        Program& operator=(Program && other) = default;
         

    }; //Program
} // FT
#endif
