/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef LOGGER_H
#define LOGGER_H

#include<iostream>
using namespace std;

namespace FT {

    namespace Util{
    
        ////////////////////////////////////////////////////////////////////////////////// Declarations
        
        /*!
         * @class Logger
         * @brief Defines a multi level static logger for feat.
         */

        class Logger
        {
            public:
                
                static Logger* initLogger();
                
                static void destroy();

                void set_log_level(int& verbosity);
                
                int get_log_level();
                
                /// print message with verbosity control. 
                string log(string m, int v, string sep="\n") const;
                
            private:
                
                int verbosity;
                
                static Logger* instance;
         
        };
        
        static Logger &logger = *Logger::initLogger();
    }
}
#endif
