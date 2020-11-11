/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "logger.h"
#include "error.h"

namespace FT {

    namespace Util{
    
        Logger* Logger::instance = NULL;
        
        Logger* Logger::initLogger()
        {
            if (!instance)
            {
                instance = new Logger();
            }

            return instance;
        }
        
        void Logger::destroy()
        {
            if (instance)
                delete instance;
                
            instance = NULL;
        }
        
        void Logger::set_log_level(int &verbosity)
        {
            if(verbosity <=3 && verbosity >=0)
                this->verbosity = verbosity;
            else
            {
                WARN("'" + std::to_string(verbosity) + "' is not a valid "
                        "verbosity. Setting to default 2\n");
                WARN("Valid Values :\n\t0 - none\n\t1 - progress\n"
                        "\t2 - minimal\n\t3 - all");
                this->verbosity = 2;
                verbosity = 2;
            }
        }
        
        int Logger::get_log_level()
        {
            return verbosity;
        }
        
        /// print message with verbosity control. 
        string Logger::log(string m, int v, string sep) const
        {
            /* prints messages based on verbosity level. */
	        string msg = "";
	
            if (verbosity >= v)
            {
                std::cout << m << sep;
                msg += m+sep;
            }
            return msg;
        }

    }

}
