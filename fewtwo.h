/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include <iostream>
#include "Eigen/Dense.h"
using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

class Fewtwo {
    /* main class for the Fewtwo learner.
    
    Fewtwo optimizes feature represenations for a given 
    machine learning algorithm. It does so by using 
    evolutionary computation to optimize a population of 
    programs. Each program represents a set of feature 
    transformations. 
    */
public : 
    int pop_size; // popsize
    int gens; // number of generations
    string selection; // 
    string ml; 
    vector<Individual> pop(); // population
    MatrixXd F; // matrix of fitness values for population
    float cross_ratio; // ratio of crossover to mutation. 
    int max_stall; // termination criterion if best model does not improve for max_stall 
                   // generations. 0 means not to use
    

    
	void Fewtwo(){
        // initialization routine.
    }
	void ~Fewtwo(){}
    
    void fit(MatrixXd& X, VectorXd& y){
        // train a model. 
    }

    VectorXd predict(MatrixXd& X){
        // predict on unseen data.
        }
    
    MatrixXd transform(MatrixXd& X, Individual ind = Individual()){
        // transform an input matrix using a program.    
    }
    
    VectorXd fit_predict(MatrixXd& X, VectorXd& y){
        // convenience function calls fit then predict.
        fit(X,y);
        return predict(X);
    }
    
    MatrixXd fit_transform(MatrixXd& X, VectorXd& y){
        // convenience function calls fit then transform. 
        fit(X,y);
        return transform(X);
    }
    

}


