#include <vector>
#include <unordered_set>
#include <iostream> 

int main()
{
    long max_id = 100000;

    std::vector<long> idx;

    for (long i =0; i < max_id; ++i)
        idx.push_back(i);

    std::unordered_set<long> idxSet(idx.begin(),idx.end());
    
    std::cout << "idxSet size: " << idxSet.size() << "\n";
    std::cout << "max_size = " << idxSet.max_size() << "\n"; 
    std::cout << "max_bucket_count = " << idxSet.max_bucket_count() << "\n";
    std::cout << "max_load_factor = " << idxSet.max_load_factor() << "\n";


    int found = 0;
    for (long i =0; i < max_id; ++i)
    {
        if(idxSet.find(i) == idxSet.end()) 
        {
            std::cout << i << " not found in idxSet. Aborting\n";
            exit(0);
        }
        else
            ++found;
    }
    std::cout << "found " << found << " of " << max_id << "\n";
}
