#include <stdio.h>
#include <fstream>
#include <streambuf>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <cstdlib>
#include <sstream>

using namespace std;

std::string ltrim(std::string str, const std::string& chars = "\t\n\v\f\r ")
{
    str.erase(0, str.find_first_not_of(chars));
    return str;
}
 
std::string rtrim(std::string str, const std::string& chars = "\t\n\v\f\r ")
{
    str.erase(str.find_last_not_of(chars) + 1);
    return str;
}
 
std::string trim(std::string str, const std::string& chars = "\t\n\v\f\r ")
{
    return ltrim(rtrim(str, chars), chars);
}

std::string vectorToString(int startIndex, std::vector<string> vec)
{
    string res ="";
    int x;
    
    for(x = startIndex; x < vec.size(); x++)
        res += vec[x]+", ";
        
    return res;
}

/*
template function to convert integers to string for logging
*/
template <typename T>
string to_string(const T& value)
{
    std::stringstream ss;
    ss << value;
    return ss.str();
}

vector<string> split(const string &s, char delim) {
    stringstream ss(s);
    string item;
    vector<string> tokens;
    while (getline(ss, item, delim)) {
        tokens.push_back(item);
    }
    return tokens;
}

int main(int argc, char *argv[])
{
    string icdFile = "./icd9_25000_caseControlStatus.txt";
    string predictFile = "./predict_25000_measures.txt";
    string bmiFile = "./covariates.csv";
    string outX = "./outputX.csv";
    string outLongitudinal = "./outputLongitudinal.csv";
    
    map<string, int> pidMap;
    map<string, int> yVal;
    map<string, bool> longitudinalPresent;
    
    string input;
    int index = 0;
    int val;
    
    ifstream inFile;
    
    //read icd file and create map of class for each pid
    inFile.open(icdFile.c_str());
    
    getline(inFile, input);
    
	while(getline(inFile, input))
	{
	    index = input.find('\t');
	    if(trim(input.substr(index)).compare("NA") != 0)
    	    yVal[trim(input.substr(0, index))] = atoi(trim(input.substr(index)).c_str());
	}
	
	inFile.close();
	
	printf("ICD file read\n");
	
	inFile.open(predictFile.c_str());
	
	ofstream outFile(outX.c_str());
	
	std::vector<std::string> result;
	bool breakFlag;	
	
	int count = 0;
	int breakCount = 0;
	
	//processing first line
	getline(inFile, input);
	result.clear();
	
	std::istringstream headerStream(input);
	for(std::string token; headerStream >> token;)        
        result.push_back(trim(token));
        
    outFile << vectorToString(1, result)+"class\n";  
    
    index = 0;  
	
	while(getline(inFile, input))
	{
	
	    count ++;
	    
	    result.clear();
	    std::istringstream stream(input);
	    
	    breakFlag = false;
	    for(std::string token; stream >> token;)
	    {
	        if(token.compare("NA") == 0)
	        {
	            breakCount++;
	            breakFlag = true;
	            break;
	        }
	        if(!token.compare("Female"))
	            token = "0";    
	        if(!token.compare("Male"))
	            token = "1";
	        if(!token.compare("Unknown"))
	            token = "2";
            result.push_back(trim(token));
        }
        
        if(!breakFlag && yVal.find(result[0]) != yVal.end())
        {
            //printf("%s", (vectorToString(1, result)+to_string(yVal[result[0]])+"\n").c_str());
            outFile << vectorToString(1, result)+to_string(yVal[result[0]])+"\n";
            
            //outFile << to_string(index)+"\n";
            
            if(pidMap.find(trim(result[0])) != pidMap.end())
                printf("Patient %s data present twice\n", trim(result[0]).c_str());
                
            pidMap[trim(result[0])] = index;
            index ++;
        }
	}
	
	inFile.close();
	outFile.close();
	
	map<string, int>::iterator itr;
	for(itr = pidMap.begin(); itr != pidMap.end(); itr++)
	    longitudinalPresent[itr->first.c_str()] = false;
	
	printf("Records read %d and records neglected are %d due to NULL values and found class mapping for %d records\n", count, breakCount, index);
	
	printf("\n\n*****Starting to parse longitudinal data\n");
	
	inFile.open(bmiFile.c_str());
	outFile.open(outLongitudinal.c_str());
	
	getline(inFile, input);
	
	while(getline(inFile, input))
	{
	    std::istringstream stream(input);
	    
	    result = split(input, ',');
	    
	    //for(string str : result)
	    //printf("String is %s\n", str.c_str());
	    
	    string pid = trim(result[1].substr(1, result[1].length()-2));
	    
	    //printf("PID is %s\n", pid.c_str());
	    if(pidMap.find(pid) != pidMap.end())
	    {
	        //printf("Found\n");
	        longitudinalPresent[pid] = true;
	        outFile << to_string(pidMap[pid])+", "+result[9]+", 0"+", BMI\n";
	    }
	    
	    //break;
	}
	
	inFile.close();
	outFile.close();
	
	int noLongitudinal = 0;
	
	map<string, bool>::iterator itr2;
	for(itr2 = longitudinalPresent.begin(); itr2 != longitudinalPresent.end(); itr2++)
	    if(itr2->second == false)
	        noLongitudinal++;
	        
    printf("Longitudinal present but no X-y %d\n", noLongitudinal);
}


